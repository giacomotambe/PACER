"""
Predittore Social-STGCNN (§1 spec, Mohamed et al. CVPR 2020) — architettura
spatio-temporal graph convolution (ST-GCNN) + time-extrapolator CNN
(TXP-CNN), kernel di adiacenza sociale = inverso della distanza (gia'
calcolato in data/windowing.py::social_adjacency).

Interfaccia richiesta dalla spec (§1): PREDICT(Y_{<=t}, k) -> (mu_{t+k|t},
Sigma_{t+k|t}). Il modello, come nell'architettura originale, produce in un
solo forward pass l'intero orizzonte k=1..H; PREDICT(.,k) e' quindi
implementata come un unico forward + slicing sull'offset k (identico al
comportamento reale della rete, non un'approssimazione).

Output per nodo/istante predetto: gaussiana bivariata (mu_x, mu_y, sigma_x,
sigma_y, rho) -> (mu, Sigma), usata direttamente come base dello score di
Mahalanobis (§2 della nota, §3 della spec).
"""
import torch
import torch.nn as nn


class STGCNNLayer(nn.Module):
    """Un layer di convoluzione grafo-temporale: convoluzione grafica
    (aggregazione via adiacenza sociale A_t, diversa per ogni istante t)
    seguita da convoluzione temporale 1D (kernel=3) sui nodi, con residuo."""

    def __init__(self, in_c, out_c, t_kernel=3):
        super().__init__()
        self.gcn_lin = nn.Linear(in_c, out_c)
        self.tcn = nn.Conv1d(out_c, out_c, kernel_size=t_kernel,
                              padding=t_kernel // 2)
        self.res = nn.Linear(in_c, out_c) if in_c != out_c else nn.Identity()
        self.act = nn.PReLU()
        self.bn = nn.BatchNorm1d(out_c)

    def forward(self, X, A):
        # X: (T,N,C_in)  A: (T,N,N)
        T, N, _ = X.shape
        agg = torch.einsum('tnm,tmc->tnc', A, X)   # aggregazione sociale per istante
        H = self.gcn_lin(agg)                       # (T,N,C_out)
        # convoluzione temporale: porta a (N, C_out, T)
        Hc = H.permute(1, 2, 0)                      # (N,C_out,T)
        Hc = self.tcn(Hc)
        Hc = self.bn(Hc)
        Hc = Hc.permute(2, 0, 1)                      # (T,N,C_out)
        res = self.res(X)
        out = self.act(Hc + res)
        return out


class TXPCNN(nn.Module):
    """Time-Extrapolator CNN: mappa T_obs=8 -> T_pred=H sull'asse temporale,
    poi proietta al numero di canali di output della gaussiana bivariata."""

    def __init__(self, in_c, hidden_c, obs_len, pred_len, out_c=5):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        # conv1d che opera sull'asse temporale (canali = feature dei nodi)
        self.expand = nn.Conv1d(obs_len, pred_len, kernel_size=3, padding=1)
        self.refine1 = nn.Conv1d(pred_len, pred_len, kernel_size=3, padding=1)
        self.refine2 = nn.Conv1d(pred_len, pred_len, kernel_size=3, padding=1)
        self.act = nn.PReLU()
        self.out_proj = nn.Linear(in_c, out_c)

    def forward(self, H):
        # H: (T_obs,N,C) -> vogliamo trattare T come "canali" della conv1d
        T, N, C = H.shape
        Hn = H.permute(1, 0, 2)          # (N, T_obs, C)
        Hp = self.expand(Hn)              # (N, T_pred, C)
        Hp = self.act(Hp + self.refine1(Hp))
        Hp = self.act(Hp + self.refine2(Hp))
        out = self.out_proj(Hp)           # (N, T_pred, out_c)
        return out.permute(1, 0, 2)        # (T_pred, N, out_c)


class SocialSTGCNN(nn.Module):
    def __init__(self, obs_len=8, pred_len=12, hidden_dim=32, n_stgcnn=2):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        layers = []
        in_c = 2
        for i in range(n_stgcnn):
            layers.append(STGCNNLayer(in_c, hidden_dim))
            in_c = hidden_dim
        self.stgcnn = nn.ModuleList(layers)
        self.txpcnn = TXPCNN(hidden_dim, hidden_dim, obs_len, pred_len, out_c=5)

    def encode(self, X, A):
        H = X
        for layer in self.stgcnn:
            H = layer(H, A)
        return H  # (T_obs, N, hidden)

    def forward(self, X, A):
        """X: (T_obs,N,2) velocita' relative osservate. A: (T_obs,N,N).
        Ritorna raw output (T_pred,N,5): mu_x,mu_y,log_sx,log_sy,rho_raw."""
        H = self.encode(X, A)
        raw = self.txpcnn(H)
        return raw

    @staticmethod
    def raw_to_gaussian(raw, min_sigma=1e-2):
        """raw: (...,5) -> mu(...,2), sigma(...,2), rho(...)"""
        mu = raw[..., 0:2]
        sx = torch.exp(raw[..., 2]).clamp(min=min_sigma)
        sy = torch.exp(raw[..., 3]).clamp(min=min_sigma)
        rho = torch.tanh(raw[..., 4]) * 0.9   # evita |rho|=1 esatto
        return mu, sx, sy, rho

    @staticmethod
    def sigma_matrix(sx, sy, rho):
        cov = sx * sy * rho
        Sigma = torch.stack([
            torch.stack([sx * sx, cov], dim=-1),
            torch.stack([cov, sy * sy], dim=-1),
        ], dim=-2)
        return Sigma


def gaussian_nll(mu, sx, sy, rho, target, mask=None):
    """NLL della gaussiana bivariata (Social-LSTM/STGCNN, standard).
    mu:(T,N,2) sx,sy,rho:(T,N) target:(T,N,2) mask:(N,) bool o None."""
    zx = (target[..., 0] - mu[..., 0]) / sx
    zy = (target[..., 1] - mu[..., 1]) / sy
    one_m_rho2 = (1 - rho ** 2).clamp(min=1e-3)
    nll = 0.5 * torch.log(one_m_rho2) + torch.log(sx) + torch.log(sy) + \
        1.0 / (2 * one_m_rho2) * (zx ** 2 + zy ** 2 - 2 * rho * zx * zy)
    nll = nll + torch.log(torch.tensor(2 * 3.141592653589793))
    if mask is not None:
        nll = nll[:, mask]
    return nll.mean()
