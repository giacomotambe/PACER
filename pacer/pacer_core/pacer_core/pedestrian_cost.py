"""Costo pedonale a gaussiane ad AREA COSTANTE — alternativa SOFT al vincolo
rigido di mpc/unicycle_nmpc.py (nota vault "Costo pedonale a gaussiane ad
area costante — alternativa soft al vincolo MPC", §2-§3-§7). Codice NUOVO:
non tocca mpc/unicycle_nmpc.py ne' experiment/mpc_ab_mission.py.

Per ogni pedone j e offset k=1..N, gaussiana (generalizzata, esponente
gamma, §7 nota) centrata in mu_jk, con covarianza Sigma_jk = (q_k/beta)^2 *
Sigma_hat_jk (§3 nota: lega il "raggio" della gaussiana al bordo
dell'ellisse TUPAC calibrata a beta deviazioni standard) e altezza h_jk
fissata dal vincolo di AREA COSTANTE sotto la superficie (§2/§7 nota):

  gamma=2 (gaussiana standard):  area = h * 2*pi*sqrt(det Sigma)
                                  h = A / (2*pi*sqrt(det Sigma))
  gamma!=2 (code piu' pesanti):  area = h * pi*sqrt(det Sigma) * C(gamma)
                                  C(gamma) = (2/gamma) * 2^(2/gamma) * Gamma(2/gamma)
                                  h = A / (pi*sqrt(det Sigma) * C(gamma))

C(2) = 2, che ridà esattamente la formula gamma=2 — verificato in
experiment/verify_pedestrian_cost.py insieme all'area vera (integrazione
numerica), non solo alla formula analitica.
"""
import numpy as np
from scipy.special import gamma as gamma_fn


def area_const_height(Sigma, A=1.0, gamma_exp=2.0):
    """h_{j,k} t.c. l'integrale su R^2 di g(p)=h*exp(-0.5*d(p)^gamma_exp) sia
    A per ogni Sigma (batch: Sigma shape (...,2,2), ritorna shape (...))."""
    det = np.linalg.det(Sigma)
    if gamma_exp == 2.0:
        return A / (2.0 * np.pi * np.sqrt(det))
    C = (2.0 / gamma_exp) * 2.0 ** (2.0 / gamma_exp) * gamma_fn(2.0 / gamma_exp)
    return A / (np.pi * np.sqrt(det) * C)


def scaled_covariance(Sigma_hat, q_k, beta):
    """Sigma_{j,k} = (q_k/beta)^2 * Sigma_hat_{j,k} (§3 nota). Sigma_hat:
    (N,2,2), q_k: (N,) -> (N,2,2)."""
    scale = (np.asarray(q_k) / beta) ** 2
    return scale[:, None, None] * Sigma_hat


def build_soft_obstacle(mu_k, Sigma_hat_k, q_k, beta, gamma_exp=2.0, A=1.0):
    """mu_k: (N,2) centri predetti (uno per passo k=1..N). Sigma_hat_k:
    (N,2,2) covarianza grezza del predittore. q_k: (N,) quantile TUPAC per
    ciascun k. Ritorna il dict consumato da
    unicycle_nmpc_soft.pedestrian_cost_term / field_value."""
    Sigma_k = scaled_covariance(Sigma_hat_k, q_k, beta)
    Sigma_inv_k = np.linalg.inv(Sigma_k)
    h_k = area_const_height(Sigma_k, A=A, gamma_exp=gamma_exp)
    return {"mu": mu_k, "Sigma": Sigma_k, "Sigma_inv": Sigma_inv_k, "h": h_k, "gamma": gamma_exp}


def gaussian_cost_value(p, mu, Sigma_inv, h, gamma_exp):
    """p, mu: (...,2) (broadcasting standard numpy). Sigma_inv: (...,2,2).
    h: (...) (stesso shape "batch" di mu/Sigma_inv). Ritorna g(p), shape
    del broadcast di p e mu."""
    diff = p - mu
    tmp = np.einsum('...i,...ij->...j', diff, Sigma_inv)
    maha2 = np.einsum('...j,...j->...', tmp, diff)
    d = np.sqrt(np.maximum(maha2, 0.0))
    return h * np.exp(-0.5 * d ** gamma_exp)


def field_value(p_grid, obstacle, reduce="sum"):
    """Valuta il campo di costo del SINGOLO pedone su una griglia di punti
    p_grid (...,2) (per ispezione/plot, §4 nota): somma o inviluppo (max)
    dei contributi di TUTTI i k, NON sincronizzati nel tempo. Da usare solo
    per visualizzazione — nel costo MPC serve il termine sincronizzato
    (unicycle_nmpc_soft.pedestrian_cost_term), mai questo campo."""
    mu, Sigma_inv, h, gamma_exp = (obstacle["mu"], obstacle["Sigma_inv"],
                                     obstacle["h"], obstacle["gamma"])
    N = mu.shape[0]
    vals = np.stack([gaussian_cost_value(p_grid, mu[k], Sigma_inv[k], h[k], gamma_exp)
                      for k in range(N)], axis=0)
    return vals.sum(axis=0) if reduce == "sum" else vals.max(axis=0)
