# models/

Mounted at `/models` in the container (`./models:/models` in both compose files).
Copy trained files here from the host; no `docker cp` needed.

| file | what it is | used by |
|---|---|---|
| `predictor_sim.pt` | Social-STGCNN (obs 8, pred 12, hidden 32), trained on open-field social-force simulation | `pacer_predictor` (`model_path`, default) |
| `cal_sim.npz` | Simulated D_cal from the closed-loop benchmark: robot with the final controller, people reacting to it, open scene (shyB) | `pacer_calibration` (`calibration_path`, default) |
| `cal_real.npz` | **Real** D_cal, created by you from Qualisys tracks with `tools/make_dcal_real.sh` (not in git) | `calibration:=/models/cal_real.npz` |

Details of `cal_sim.npz`:
- 59,857 windows, with Mondrian pools by distance from the robot (near = d < 2.5 m).
- 4,104 of the k=1 scores are in the near group.
- With the `standard` preset, the initial q for near pedestrians at k=1 is 4.76, the same as the
  benchmark (v50).

`cal_sim.npz` is fine for the first tests. For the experiments and the paper, use a
D_cal collected on the real robot moving among people: the guarantee assumes that
D_cal and deployment come from the same distribution.

Other files in this folder (your own models and D_cals) are ignored by git.
