"""Prova della MPC v2 compilata (JIT) sul robot (rev. 2026-11-18, nota PACER §3s).

Risolve gli stessi 180 problemi registrati nel benchmark (crossing, sidewalk, open con 20 persone,
`mpc_problems_sample.pkl`) con il solutore interpretato e con quello compilato, e confronta tempi,
esito e primo comando. La prima esecuzione compila 11 librerie (alcuni minuti); le successive
le ricaricano dalla cache (~/.cache/pacer_mpc_jit).

Uso (sul robot, nel workspace compilato):
    python3 tools/mpc_jit_check.py                 # gcc -O1
    PACER_JIT_FLAGS=-O2 python3 tools/mpc_jit_check.py
Richiede: gcc, casadi (pip install casadi), numpy.
"""
import importlib.util
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MOD = os.path.join(HERE, "..", "pacer", "pacer_core", "pacer_core", "unicycle_nmpc_v2.py")


def load(name):
    spec = importlib.util.spec_from_file_location(name, MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run(m, prs):
    out = []
    for pr in prs:
        args = (pr["x0"], pr["v_mis"], pr["w_mis"], pr["goal"], pr["obstacles"], pr["N"], pr["v_max"],
                pr["omega_max"])
        kw = dict(u_warm=pr["u_warm"], path_dir=pr["path_dir"], path_org=pr["path_org"],
                  halfplanes=pr["halfplanes"], statics=pr["statics"])
        m.solve_step(*args, **kw)                    # costruisce la taglia se manca
        t0 = time.perf_counter()
        u0, ok, uf, mg, st = m.solve_step(*args, **kw)
        out.append((time.perf_counter() - t0, st, np.asarray(u0, float)))
    return out


def main():
    prs = pickle.load(open(os.path.join(HERE, "mpc_problems_sample.pkl"), "rb"))
    N = prs[0]["N"]
    interp = load("nmpc_interp")
    interp.configure(JIT=False)
    jit = load("nmpc_jit")
    jit.configure(JIT=True)
    t0 = time.perf_counter()
    jit.precompile(N)
    print(f"precompilazione JIT: {time.perf_counter() - t0:.0f} s (0 se gia' in cache)")
    if jit._JIT_FAILED:
        print("JIT FALLITO:", jit._JIT_FAILED[0][:500])
        sys.exit(1)
    interp.precompile(N)
    ri, rj = run(interp, prs), run(jit, prs)
    for sc in ("crossing", "sidewalk", "open", None):
        idx = [i for i, p in enumerate(prs) if sc is None or p["scene"] == sc]
        for lab, r in (("interpretato", ri), ("JIT", rj)):
            t = np.array([r[i][0] for i in idx]) * 1e3
            print(f"{sc or 'tutte':9s} {lab:12s} media {t.mean():6.1f} ms  p95 {np.percentile(t, 95):6.1f}  "
                  f"max {t.max():6.1f}  ottimo {np.mean([r[i][1] == 'optimal' for i in idx]) * 100:5.1f}%")
    same = np.mean([ri[i][1] == rj[i][1] for i in range(len(prs))])
    du = max(float(np.abs(ri[i][2] - rj[i][2]).max()) for i in range(len(prs)) if ri[i][1] == rj[i][1])
    print(f"stesso esito {same * 100:.1f}%, max |u0_JIT - u0_interp| a parita' di esito {du:.2e}")
    print(f"speedup medio {np.mean([x[0] for x in ri]) / np.mean([x[0] for x in rj]):.2f}x")


if __name__ == "__main__":
    main()
