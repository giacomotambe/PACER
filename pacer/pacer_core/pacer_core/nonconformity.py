"""Score di non-conformita' di Mahalanobis (§3 della spec, §2 della nota)."""
import numpy as np


def mahalanobis_score(Y, mu, Sigma, eps=1e-6):
    """Y,mu: (2,)  Sigma: (2,2). Ritorna scalare R = ||Y-mu||_{Sigma^-1}."""
    diff = Y - mu
    Sigma_reg = Sigma + eps * np.eye(2)
    Sinv = np.linalg.inv(Sigma_reg)
    return float(np.sqrt(max(diff @ Sinv @ diff, 0.0)))


def ellipse_area(Sigma, q, eps=1e-6):
    """Area dell'ellisse {y : (y-mu)^T Sigma^-1 (y-mu) <= q^2} = pi*q^2*sqrt(det Sigma)."""
    Sigma_reg = Sigma + eps * np.eye(2)
    det = max(np.linalg.det(Sigma_reg), 0.0)
    if not np.isfinite(q):
        return np.inf
    return float(np.pi * (q ** 2) * np.sqrt(det))
