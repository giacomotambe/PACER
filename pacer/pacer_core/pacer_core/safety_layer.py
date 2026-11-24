"""Strato di sicurezza deterministico (rev. 2026-11-17, nota PACER §3q-§3r), stessa
matematica di 13_closed_loop_benchmark/common.py (braking_path, estop_cv_violators).

Passive safety (Fraichard 2007): il robot deve potersi sempre fermare prima di toccare una
persona; se c'e' contatto, il robot e' gia' fermo. Dopo aver applicato il comando (v, w) per
t_cmd secondi, il robot frena in linea retta con decelerazione a_dec. Una persona prosegue a
velocita' costante con deviazione fino a a_ped*tau^2/2. Se esiste un istante, prima
dell'arresto, in cui la distanza scende sotto clearance_l + margin + deviazione, il comando
non e' sicuro e va sostituito da una frenata.

Ipotesi da dichiarare nel paper: a_ped (0 = velocita' costante), latenza <= t_cmd,
percezione affidabile (posizione e velocita' delle persone).
"""
import numpy as np


def braking_path(x, v, w, t_cmd, a_dec, taus):
    """Posizioni (len(taus), 2) e velocita' del robot ai tempi taus: comando (v, w) per t_cmd,
    poi frenata rettilinea con decelerazione a_dec. x = (px, py, theta)."""
    taus = np.asarray(taus, float)
    t1 = np.minimum(taus, t_cmd)
    th = x[2] + w * t1
    out = np.zeros((len(taus), 2))
    if abs(w) > 1e-6:
        out[:, 0] = x[0] + v / w * (np.sin(x[2] + w * t1) - np.sin(x[2]))
        out[:, 1] = x[1] - v / w * (np.cos(x[2] + w * t1) - np.cos(x[2]))
    else:
        out[:, 0] = x[0] + v * t1 * np.cos(x[2])
        out[:, 1] = x[1] + v * t1 * np.sin(x[2])
    tb = np.maximum(taus - t_cmd, 0.0)
    tb_c = np.minimum(tb, v / a_dec)
    s = v * tb_c - a_dec * tb_c ** 2 / 2
    out[:, 0] += s * np.cos(th)
    out[:, 1] += s * np.sin(th)
    spd = np.where(taus <= t_cmd, v, np.maximum(v - a_dec * tb, 0.0))
    return out, spd


def unsafe_people(x, v, w, ped_pos, ped_vel, t_cmd=0.4, a_dec=1.5, a_ped=0.0,
                  clearance_l=0.6, margin=0.1, step=0.05, away_only=False):
    """Indici delle persone che il robot non riuscirebbe a evitare frenando dopo il comando
    (v, w). ped_pos, ped_vel: (n, 2). Lista vuota = comando sicuro."""
    ped_pos = np.asarray(ped_pos, float).reshape(-1, 2)
    ped_vel = np.asarray(ped_vel, float).reshape(-1, 2)
    if v <= 1e-3 or len(ped_pos) == 0:
        return []
    T = t_cmd + v / a_dec
    taus = np.arange(step, T + 1e-9, step)
    r, spd = braking_path(x, v, w, t_cmd, a_dec, taus)
    moving = spd > 1e-3
    taus, r = taus[moving], r[moving]
    if len(taus) == 0:
        return []
    p = ped_pos[None] + taus[:, None, None] * ped_vel[None]
    d = np.linalg.norm(p - r[:, None, :], axis=2)
    thr = clearance_l + margin + a_ped * taus[:, None] ** 2 / 2
    bad = d < thr
    if away_only:
        # rev. 2026-11-22 (nota §3s): conta solo se il moto del robot porta piu' vicino alla persona
        # di quanto ci arriverebbe lei con il robot fermo. Senza questo, una persona ferma (o che
        # spinge verso il robot) entro clearance_l + margin lo blocca per sempre (deadlock visto nel
        # benchmark, open seed 1010006); se c'e' contatto, e' la persona a venire addosso.
        d_stay = np.linalg.norm(p - np.asarray(x[:2], float)[None, None], axis=2)
        bad &= d < d_stay - 1e-6
    return [int(i) for i in np.nonzero(bad.any(axis=0))[0]]


def brake_command(v_last, dt_since, a_dec, w_in=0.0, rotate_when_stopped=False):
    """Frenata d'emergenza: velocita' ridotta di a_dec*dt rispetto all'ultimo comando inviato,
    velocita' angolare zero. Con rotate_when_stopped (rev. 2026-11-22) a robot fermo si lascia
    passare w_in: la rotazione sul posto non sposta il disco del robot."""
    v = max(0.0, v_last - a_dec * max(dt_since, 0.0))
    return v, (float(w_in) if (rotate_when_stopped and v <= 1e-6) else 0.0)
