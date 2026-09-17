"""
QUANTILE_TUPAC (Teorema 2.3, §12.5 nota / §4 spec) e le due baseline di
confronto (§5 spec):
  - split-conformal statico: quantile fisso calcolato una sola volta su
    D_cal, mai aggiornato durante lo stream (Lemma 1 stile Lindemann et al.).
  - TUC (Teorema 2.2): stessa costruzione time-uniform ma con correzione
    Hoeffding invece della divergenza KL di Bernoulli psi. La spec la cita
    solo per nome ("riferimento intermedio"), senza riportarne la formula
    esatta: qui si implementa l'analogo Hoeffding standard della stessa
    costruzione a budget u_s, come baseline di confronto onesta — vedi nota
    esplicita nel report tecnico su questa scelta implementativa.

Scelte di h(t) (§12.4 nota / punto 4 §5 spec), entrambe PMF valide su N:
  - "heavy_tail":  h(t) = 1/(t(t+1)), t=1,2,...  (somma telescopica ESATTA
    = 1, nessuna assunzione sulla durata dello stream).
  - "peaked":      Poisson(lambda=mode) troncata e rinormalizzata su un
    supporto finito — piccata su una durata di missione attesa.
"""
import numpy as np
from scipy.stats import poisson


def psi_bernoulli_kl(x, p, eps=1e-12):
    """psi(x,p) = p*log(p/x) + (1-p)*log((1-p)/(1-x))  (KL Bernoulli(p)||Bernoulli(x))."""
    x = np.clip(x, eps, 1 - eps)
    p = np.clip(p, eps, 1 - eps)
    return p * np.log(p / x) + (1 - p) * np.log((1 - p) / (1 - x))


class HFunction:
    """PMF h su {1,2,...} con log h(t) e cumulata Sigma_{r=0}^{t0} h(r)
    calcolabili in forma chiusa/stabile per t grandi."""

    def __init__(self, kind="heavy_tail", mode=150, T_support=200000):
        self.kind = kind
        self.mode = mode
        if kind == "heavy_tail":
            pass  # h(t) = 1/(t(t+1)), forma chiusa
        elif kind == "peaked":
            ks = np.arange(1, T_support + 1)
            logpmf = poisson.logpmf(ks, mu=mode)
            m = logpmf.max()
            w = np.exp(logpmf - m)
            Z = w.sum()
            self._log_norm_const = m + np.log(Z)   # log Z_tot t.c. somma pmf normalizzata = 1
            self._T_support = T_support
        else:
            raise ValueError(kind)

    def log_h(self, s):
        s = np.asarray(s, dtype=float)
        if self.kind == "heavy_tail":
            out = -np.log(s) - np.log(s + 1)
            out = np.where(s >= 1, out, -np.inf)
            return out
        else:
            out = poisson.logpmf(s, mu=self.mode) - self._log_norm_const
            out = np.where((s >= 1) & (s <= self._T_support), out, -np.inf)
            return out

    def cum_sum_0_to(self, t0):
        """Sigma_{r=0}^{t0} h(r), h(0):=0 per convenzione (supporto t>=1)."""
        if t0 < 1:
            return 0.0
        if self.kind == "heavy_tail":
            return 1.0 - 1.0 / (t0 + 1)   # telescopica esatta
        else:
            ks = np.arange(1, min(t0, self._T_support) + 1)
            return float(np.exp(poisson.logpmf(ks, mu=self.mode) - self._log_norm_const).sum())


def _u_s(s, delta, h: HFunction, t0=0):
    cum = h.cum_sum_0_to(t0)
    log_hs = h.log_h(s)
    if not np.isfinite(log_hs):
        return np.inf
    num = np.log((1.0 / delta) * max(1.0 - cum, 1e-300))
    return (num - log_hs) / (s + 1)


def quantile_from_sorted_pool(pool_sorted, s, alpha, delta, h: HFunction, t0=0,
                                bound="kl"):
    """pool_sorted: array ordinato crescente, len==s. Ritorna q_hat (float, puo'
    essere +inf se nessun elemento del pool soddisfa le condizioni — burn-in
    insufficiente o h scelta male, §12.6/punto 4 §5)."""
    if s == 0:
        return np.inf
    u_s = _u_s(s, delta, h, t0=t0)
    ranks = np.arange(1, s + 1)          # F_hat al rango i = i/s
    F = ranks / s
    p = (s / (s + 1.0)) * F
    cond2 = ranks >= (1 - alpha) * (s + 1)
    if bound == "kl":
        psi = psi_bernoulli_kl(1 - alpha, p)
        cond1 = psi >= u_s
    else:  # Hoeffding (TUC, baseline di confronto approssimata — vedi docstring modulo)
        eps = np.sqrt(np.maximum(u_s, 0.0) / 2.0)
        cond1 = F >= (1 - alpha) + eps
    ok = cond1 & cond2
    idx = np.argmax(ok)
    if not ok[idx]:
        return np.inf
    return float(pool_sorted[idx])


def static_split_conformal_quantile(pool_sorted, alpha):
    """Quantile fisso stile Lindemann/Vovk: ceil((1-alpha)(n+1))-esima
    statistica d'ordine di D_cal, mai piu' aggiornato (baseline §5)."""
    n = len(pool_sorted)
    if n == 0:
        return np.inf
    k = int(np.ceil((1 - alpha) * (n + 1)))
    k = min(max(k, 1), n)
    return float(pool_sorted[k - 1])
