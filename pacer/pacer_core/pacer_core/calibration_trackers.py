"""Copia di 13_closed_loop_benchmark/trackers.py (rev. 2026-11-04), stessa
logica validata nel benchmark closed-loop. Calibratori per offset del benchmark closed-loop.

Interfaccia comune, per UN offset k:
    tr.q()                -> quantile da usare ORA (issue time t)
    tr.update(t_issue, scores)  -> batch di score maturati, emessi a t_issue

Metodi:
  none    q = 0: MPC senza CP (solo media predetta + margine L)
  static  split-conformal fisso su D_cal (Lindemann et al.)
  aci     adaptive conformal inference (Gibbs-Candes, come Dixit et al.)
  kl      KL-TUPAC, interlacciato per fase t mod k, un pedone a caso per
          batch (rev. 2026-11-03, Sez. IV-B/C del paper)
  bet     betting con media di batch su griglia, interlacciato per fase
  fbet    bet con capitale limitato e puntata fissa (rev. 2026-11-23, da trackers_fast.py del benchmark)
  xbet    bet a miscela di 10 puntate fisse, ognuna a capitale limitato: standard dal 2026-11-23
          (run v50, nota PACER §3s: q vicini k=1 -6/8% rispetto a fbet, miss sotto alpha_1 anche sotto stress)
"""
import bisect
import math
import numpy as np

from .quantile_tupac import HFunction, quantile_from_sorted_pool, static_split_conformal_quantile


class Zero:
    def __init__(self, *a, **kw):
        pass

    def q(self):
        return 0.0

    def update(self, t_issue, scores):
        pass


class Static:
    def __init__(self, dcal, alpha, **kw):
        self._q = static_split_conformal_quantile(np.sort(dcal), alpha)

    def q(self):
        return self._q

    def update(self, t_issue, scores):
        pass


class ACI:
    """alpha_{t+1} = alpha_t + gamma (alpha - err_t), q = quantile empirico
    (1-alpha_t) di D_cal + score online (Gibbs & Candes 2021)."""

    def __init__(self, dcal, alpha, gamma=0.005, **kw):
        self.alpha, self.gamma = alpha, gamma
        self.a = alpha
        self.pool = sorted(map(float, dcal))
        self._q = self._quant()

    def _quant(self):
        if self.a <= 0:
            return self.pool[-1]
        if self.a >= 1:
            return 0.0
        i = min(len(self.pool) - 1, int(math.ceil((1 - self.a) * (len(self.pool) + 1))) - 1)
        return self.pool[max(i, 0)]

    def q(self):
        return self._q

    def update(self, t_issue, scores):
        for r in scores:
            err = 1.0 if r > self._q else 0.0
            self.a += self.gamma * (self.alpha - err)
            bisect.insort(self.pool, float(r))
        self._q = self._quant()


class _KLPhase:
    def __init__(self, dcal, alpha, delta, h):
        self.pool = sorted(map(float, dcal))
        self.alpha, self.delta, self.h = alpha, delta, h
        self._q = self._recompute()

    def _recompute(self):
        return quantile_from_sorted_pool(np.asarray(self.pool), len(self.pool),
                                         self.alpha, self.delta, self.h, t0=0, bound="kl")

    def add(self, r):
        bisect.insort(self.pool, float(r))
        self._q = self._recompute()


class _BetPhase:
    """K_t(r) su griglia, fattore (1 + lambda (Xbar(r) - (1-alpha)))."""

    def __init__(self, dcal, alpha, delta, grid, eps=1e-3):
        self.alpha, self.grid = alpha, grid
        self.lam_bound = 1 / (1 - alpha) - eps
        self.log_target = math.log(1 / delta)
        self.logK = np.zeros(len(grid))
        self.y_sum, self.y_n = 1.0, 2.0
        self._q = math.inf
        # warm start: eventi a score singolo, in ordine CASUALE. Il pool D_cal
        # e' salvato ordinato, e un ordine monotono e' avversariale per il
        # betting (lambda si adatta alla sequenza): con score crescenti il
        # quantile collassa vicino a 0.
        for r in np.random.default_rng(12345).permutation(np.asarray(dcal, dtype=float)):
            self.add_batch(np.array([r]))

    def add_batch(self, rs):
        a = self.alpha
        lam = min(max((self.y_sum / self.y_n - (1 - a)) / (a * (1 - a)), 0.0), self.lam_bound)
        self.y_sum += float(np.mean(rs <= self._q))
        self.y_n += 1.0
        # Xbar(grid_i) = frazione del batch con R <= grid_i
        xbar = np.searchsorted(np.sort(rs), self.grid, side="right") / len(rs)
        self.logK += np.log1p(lam * (xbar - (1 - a)))
        ok = np.nonzero(self.logK >= self.log_target)[0]
        self._q = float(self.grid[ok[0]]) if len(ok) else math.inf


# ---- rev. 2026-11-23: varianti a capitale limitato (copia di 13_closed_loop_benchmark/trackers_fast.py) ----
CAP = 1.0      # [nats] margine del capitale sopra log(1/delta)
LAM = 0.5      # fbet: puntata fissa, frazione di lam_bound


class _FastBetPhase(_BetPhase):
    """fbet: K_{t+1}(r) = min(K_t(r) (1 + lam (Xbar(r) - (1-alpha))), C), lam fisso.
    min(K f, C) con E[f] <= 1 sotto H0(r) resta una supermartingala non negativa (Ville vale ancora);
    il limite C = exp(log(1/delta) + CAP) fa risalire il quantile dopo pochi errori."""

    def __init__(self, dcal, alpha, delta, grid, eps=1e-3, cap=CAP, lam=LAM):
        self.log_cap = math.log(1 / delta) + cap
        self.lam_frac = lam
        super().__init__(dcal, alpha, delta, grid, eps)

    def add_batch(self, rs):
        a = self.alpha
        lam = self.lam_frac * self.lam_bound
        xbar = np.searchsorted(np.sort(rs), self.grid, side="right") / len(rs)
        self.logK = np.minimum(self.logK + np.log1p(lam * (xbar - (1 - a))), self.log_cap)
        ok = np.nonzero(self.logK >= self.log_target)[0]
        self._q = float(self.grid[ok[0]]) if len(ok) else math.inf


class _MixBetPhase(_BetPhase):
    """xbet: K(r) = media_j K_j(r), K_j(r) = min(K_j(r) (1 + lam_j (Xbar(r) - (1-alpha))), C_j), lam_j fissi.
    Ogni K_j e' una supermartingala non negativa e monotona in r; la media lo e' ancora, quindi
    q = min{r: K(r) >= 1/delta} ha la stessa garanzia anytime di bet/fbet."""

    LAMS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9)

    def __init__(self, dcal, alpha, delta, grid, eps=1e-3, cap=CAP):
        self.lams = (1 / (1 - alpha) - eps) * np.array(self.LAMS)
        self.log_cap = math.log(1 / delta) + cap + math.log(len(self.LAMS))   # cap per componente
        self.logKj = np.zeros((len(self.LAMS), len(grid)))
        super().__init__(dcal, alpha, delta, grid, eps)

    def add_batch(self, rs):
        a = self.alpha
        xbar = np.searchsorted(np.sort(rs), self.grid, side="right") / len(rs)
        self.logKj = np.minimum(self.logKj + np.log1p(self.lams[:, None] * (xbar - (1 - a))[None, :]),
                                self.log_cap)
        m = self.logKj.max(0)
        self.logK = m + np.log(np.mean(np.exp(self.logKj - m), 0))
        ok = np.nonzero(self.logK >= self.log_target)[0]
        self._q = float(self.grid[ok[0]]) if len(ok) else math.inf


class Interleaved:
    """k tracker per fase phi = t_issue mod k, confidenza delta/(N_MPC k);
    quantile usato = minimo sulle fasi (Sez. IV-B)."""

    def __init__(self, kind, dcal, alpha, delta, k, n_offsets, rng, grid=None):
        self.k, self.kind, self.rng = k, kind, rng
        d = delta / (n_offsets * k)
        if kind == "kl":
            h = HFunction("heavy_tail")
            self.ph = [_KLPhase(dcal, alpha, d, h) for _ in range(k)]
        else:
            first = {"bet": _BetPhase, "fbet": _FastBetPhase, "xbet": _MixBetPhase}[kind](dcal, alpha, d, grid)
            self.ph = [first] + [_clone_bet(first) for _ in range(k - 1)]

    def q(self):
        return min(p._q for p in self.ph)

    def update(self, t_issue, scores):
        if not len(scores):
            return
        p = self.ph[t_issue % self.k]
        if self.kind == "kl":
            p.add(scores[self.rng.integers(len(scores))])   # un pedone a caso
        else:
            p.add_batch(np.asarray(scores, dtype=float))


def _clone_bet(b):
    c = object.__new__(type(b))
    c.__dict__ = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in b.__dict__.items()}
    return c


def make_tracker(method, dcal, k, n_offsets, alpha_k, delta, rng, grid=None):
    if method == "none":
        return Zero()
    if method == "static":
        return Static(dcal, alpha_k)
    if method == "aci":
        return ACI(dcal, alpha_k)
    if method in ("kl", "bet", "fbet", "xbet"):
        return Interleaved(method, dcal, alpha_k, delta, k, n_offsets, rng, grid=grid)
    raise ValueError(method)
