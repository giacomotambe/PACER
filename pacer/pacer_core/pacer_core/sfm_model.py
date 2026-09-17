"""
Social Force Model (Helbing & Molnar 1995), §2 della spec.

    m_i dv_i/dt = f_i^0 + sum_j f_ij + sum_W f_iW

Formulazione "per unita' di massa" (nessuna massa esplicita, A/B in
accelerazione/metri) — standard nelle implementazioni moderne (es.
PySocialForce), coerente con l'uso qui: le costanti sono note e fissate
dall'analista (§2 spec), non stimate da dati.

Quattro scenari (§1 report + confronto scenari), tutti costruiti sullo
stesso motore fisico (forza motrice + repulsione pedone-pedone + repulsione
da muri), differiscono solo per come vengono generate posizione iniziale e
obiettivo di ciascun pedone:

- "corridor":       corridoio bidirezionale (originale, §1 report). Incontri
                     quasi sempre frontali in uno spazio stretto -> deviazioni
                     tardive e brusche, score di Mahalanobis bimodale con rari
                     outlier enormi (osservato, vedi report §5/§8/§11).
- "corridor_soft":  STESSA geometria, ma repulsione pedone-pedone piu' dolce
                     (B piu' grande = raggio d'azione maggiore, A piu' piccolo
                     = intensita' minore) -> i pedoni iniziano a scartare
                     prima e piu' gradualmente, dovrebbe smussare la coda.
- "cross":           due flussi ortogonali che si incrociano al centro di
                     un'area quadrata (bande, non tutta l'area) -> stessa
                     logica "direzione fissa" del corridoio ma con eterogeneita'
                     di heading (orizzontale/verticale), interazioni visibili
                     da piu' lontano (l'avvicinamento all'incrocio e' graduale).
- "open_field":      posizione iniziale e obiettivo (punto, non direzione)
                     casuali in un'area quadrata, muri su tutti e 4 i lati.
                     Forza motrice ricalcolata ad ogni passo verso il punto
                     obiettivo corrente (goal-seeking dinamico, non piu'
                     direzione fissa) -> massima eterogeneita' di heading e
                     di geometria d'incontro, conflitti risolti con piu'
                     margine (area aperta, non un corridoio stretto).
"""
import numpy as np


def _driving_force(vel, e_goal, v0, tau):
    return (v0[:, None] * e_goal - vel) / tau[:, None]


def _pairwise_repulsion(pos, radius, A, B):
    """f_ij per tutte le coppie i!=j. pos: (N,2). Ritorna (N,2)."""
    N = pos.shape[0]
    if N < 2:
        return np.zeros_like(pos)
    diff = pos[:, None, :] - pos[None, :, :]          # (N,N,2) r_i - r_j
    dist = np.sqrt((diff ** 2).sum(-1)) + 1e-9          # (N,N)
    n_ij = diff / dist[:, :, None]
    r_sum = radius[:, None] + radius[None, :]             # (N,N)
    mag = A * np.exp((r_sum - dist) / B)                    # (N,N)
    np.fill_diagonal(mag, 0.0)
    f = (mag[:, :, None] * n_ij).sum(axis=1)                  # (N,2)
    return f


def _corridor_wall_repulsion(pos, radius, width, A_wall, B_wall):
    """Muri rettilinei a y=0 e y=width (semi-infiniti in x, corridoio)."""
    y = pos[:, 1]
    d_bottom = y
    d_top = width - y
    f = np.zeros_like(pos)
    mag_b = A_wall * np.exp((radius - d_bottom) / B_wall)
    mag_t = A_wall * np.exp((radius - d_top) / B_wall)
    f[:, 1] += mag_b
    f[:, 1] -= mag_t
    return f


def _box_wall_repulsion(pos, radius, xmin, xmax, ymin, ymax, A_wall, B_wall):
    """Muri rettilinei sui 4 lati di un'area rettangolare (cross/open_field)."""
    x, y = pos[:, 0], pos[:, 1]
    f = np.zeros_like(pos)
    f[:, 0] += A_wall * np.exp((radius - (x - xmin)) / B_wall)
    f[:, 0] -= A_wall * np.exp((radius - (xmax - x)) / B_wall)
    f[:, 1] += A_wall * np.exp((radius - (y - ymin)) / B_wall)
    f[:, 1] -= A_wall * np.exp((radius - (ymax - y)) / B_wall)
    return f


def _setup_corridor(n_peds, cfg, rng):
    L, W = cfg["corridor_length"], cfg["corridor_width"]
    radius = cfg["radius"]
    direction = np.where(rng.random(n_peds) < 0.5, 1.0, -1.0)
    x0 = np.where(direction > 0, 0.0, L) + rng.normal(0, 0.5, n_peds)
    y0 = rng.uniform(radius * 1.5, W - radius * 1.5, n_peds)
    pos = np.stack([x0, y0], axis=1)
    e_goal = np.stack([direction, np.zeros(n_peds)], axis=1)
    goal_x = np.where(direction > 0, L + 2.0, -2.0)

    def reached_fn(pos_t):
        return (direction > 0) & (pos_t[:, 0] >= goal_x) | (direction < 0) & (pos_t[:, 0] <= goal_x)

    def wall_fn(pos_t, rad):
        return _corridor_wall_repulsion(pos_t, rad, W, cfg["A_wall"], cfg["B_wall"])

    return pos, e_goal, None, reached_fn, wall_fn, False


def _setup_cross(n_peds, cfg, rng):
    """Due flussi ortogonali (bande) che si incrociano al centro di un'area
    quadrata di lato `field_size`. Ogni pedone appartiene al flusso
    orizzontale o verticale (heading fisso, come nel corridoio) — ma le due
    orientazioni convivono nello stesso run, dando eterogeneita' di heading
    (assente nel corridoio puro, §8/§11 report)."""
    S = cfg["field_size"]
    band = cfg["corridor_width"]
    radius = cfg["radius"]
    horiz = rng.random(n_peds) < 0.5
    direction = np.where(rng.random(n_peds) < 0.5, 1.0, -1.0)
    x0 = np.zeros(n_peds); y0 = np.zeros(n_peds)
    e_goal = np.zeros((n_peds, 2))
    goal_coord = np.zeros(n_peds)
    center = S / 2.0
    for i in range(n_peds):
        off = rng.uniform(radius * 1.5, band - radius * 1.5) - band / 2.0
        if horiz[i]:
            x0[i] = 0.0 if direction[i] > 0 else S
            x0[i] += rng.normal(0, 0.5)
            y0[i] = center + off
            e_goal[i] = [direction[i], 0.0]
            goal_coord[i] = S + 2.0 if direction[i] > 0 else -2.0
        else:
            y0[i] = 0.0 if direction[i] > 0 else S
            y0[i] += rng.normal(0, 0.5)
            x0[i] = center + off
            e_goal[i] = [0.0, direction[i]]
            goal_coord[i] = S + 2.0 if direction[i] > 0 else -2.0
    pos = np.stack([x0, y0], axis=1)

    def reached_fn(pos_t):
        coord = np.where(horiz, pos_t[:, 0], pos_t[:, 1])
        return (direction > 0) & (coord >= goal_coord) | (direction < 0) & (coord <= goal_coord)

    def wall_fn(pos_t, rad):
        return _box_wall_repulsion(pos_t, rad, -1.0, S + 1.0, -1.0, S + 1.0,
                                    cfg["A_wall"], cfg["B_wall"])

    return pos, e_goal, None, reached_fn, wall_fn, False


def _setup_open_field(n_peds, cfg, rng):
    """Posizione iniziale e obiettivo (punto) casuali in un'area quadrata di
    lato `field_size`, distanza minima imposta perche' il tragitto sia
    significativo. Forza motrice goal-seeking dinamica (ricalcolata ogni
    passo verso il punto obiettivo, non una direzione fissa come negli altri
    scenari) -> massima eterogeneita' di heading, incontri risolti con piu'
    margine di manovra (§ raccomandazione discussione)."""
    S = cfg["field_size"]
    radius = cfg["radius"]
    pos = rng.uniform(radius * 1.5, S - radius * 1.5, size=(n_peds, 2))
    goal = np.zeros((n_peds, 2))
    for i in range(n_peds):
        while True:
            g = rng.uniform(radius * 1.5, S - radius * 1.5, size=2)
            if np.linalg.norm(g - pos[i]) >= S * 0.5:
                goal[i] = g
                break

    def reached_fn(pos_t):
        return np.linalg.norm(pos_t - goal, axis=1) <= 0.4

    def wall_fn(pos_t, rad):
        return _box_wall_repulsion(pos_t, rad, 0.0, S, 0.0, S, cfg["A_wall"], cfg["B_wall"])

    return pos, None, goal, reached_fn, wall_fn, True


def _setup_corridor_groups(n_peds, cfg, rng):
    """Variante del corridoio con pedoni organizzati in GRUPPI che si
    muovono insieme (idea discussa: dipendenza *strutturale/persistente*
    tra traiettorie, non solo incontri casuali momentanei come in §5/§12).
    Ogni gruppo (dimensione `group_size`, default 3): stessa direzione,
    stessa "corsia" (centro y) con piccolo offset laterale per stare
    affiancati, stessa velocita' desiderata v0 (sincronizzata in
    simulate_run) e una forza di coesione (vedi _group_cohesion_force) che
    li tiene vicini anche sotto la repulsione da altri pedoni/gruppi."""
    L, W = cfg["corridor_length"], cfg["corridor_width"]
    radius = cfg["radius"]
    group_size = cfg.get("group_size", 3)
    n_groups = int(np.ceil(n_peds / group_size))
    group_id = np.zeros(n_peds, dtype=int)
    direction = np.zeros(n_peds)
    x0 = np.zeros(n_peds)
    y0 = np.zeros(n_peds)
    idx = 0
    for g in range(n_groups):
        size = min(group_size, n_peds - idx)
        d = 1.0 if rng.random() < 0.5 else -1.0
        lane_y = rng.uniform(radius * 1.5, W - radius * 1.5)
        base_x = 0.0 if d > 0 else L
        for j in range(size):
            i = idx + j
            group_id[i] = g
            direction[i] = d
            lateral = (j - (size - 1) / 2.0) * (radius * 2.2)   # affiancati, no overlap allo spawn
            y0[i] = np.clip(lane_y + lateral, radius * 1.5, W - radius * 1.5)
            x0[i] = base_x + rng.normal(0, 0.3)
        idx += size
    pos = np.stack([x0, y0], axis=1)
    e_goal = np.stack([direction, np.zeros(n_peds)], axis=1)
    goal_x = np.where(direction > 0, L + 2.0, -2.0)

    def reached_fn(pos_t):
        return (direction > 0) & (pos_t[:, 0] >= goal_x) | (direction < 0) & (pos_t[:, 0] <= goal_x)

    def wall_fn(pos_t, rad):
        return _corridor_wall_repulsion(pos_t, rad, W, cfg["A_wall"], cfg["B_wall"])

    return pos, e_goal, None, reached_fn, wall_fn, False, group_id


def _group_cohesion_force(pos, group_id, k_coh, max_mag):
    """Forza di coesione: ciascun membro di un gruppo e' attratto verso il
    centroide degli ALTRI membri del proprio gruppo (esclude se stesso),
    magnitudine limitata per stabilita' numerica. group_id=-1 o gruppi di
    dimensione 1 => nessuna forza (pedone solo)."""
    N = pos.shape[0]
    force = np.zeros((N, 2))
    if group_id is None:
        return force
    for g in np.unique(group_id):
        idx = np.where(group_id == g)[0]
        if len(idx) <= 1:
            continue
        centroid = pos[idx].mean(axis=0)
        for i in idx:
            other_centroid = (centroid * len(idx) - pos[i]) / (len(idx) - 1)
            diff = other_centroid - pos[i]
            f = k_coh * diff
            norm = np.linalg.norm(f)
            if norm > max_mag:
                f = f / norm * max_mag
            force[i] = f
    return force


def _setup_corridor_two_groups(n_peds, cfg, rng):
    """Scontro frontale tra ESATTAMENTE 2 gruppi coesi, direzioni opposte
    FORZATE (non estratte a caso come in _setup_corridor_groups) — risolve
    il confondimento di §14 (dove group_size grande faceva perdere la
    bidirezionalita', "raddolcendo" lo scenario): qui il conflitto frontale
    c'e' SEMPRE, per costruzione, qualunque sia la dimensione dei due
    gruppi. n_peds e' diviso a meta' tra i due gruppi (gruppo A -> destra,
    gruppo B -> sinistra); ciascun gruppo e' disposto su piu' file (non solo
    affiancati su una riga, che non ci starebbe oltre ~9 persone nei 6 m di
    larghezza) vicino al proprio estremo del corridoio, cosi' da formare un
    blocco compatto che avanza e si scontra con l'altro blocco al centro."""
    L, W = cfg["corridor_length"], cfg["corridor_width"]
    radius = cfg["radius"]
    spacing = radius * 2.3
    nA = n_peds // 2
    nB = n_peds - nA
    group_id = np.zeros(n_peds, dtype=int)
    direction = np.zeros(n_peds)
    x0 = np.zeros(n_peds)
    y0 = np.zeros(n_peds)

    def place_block(count, base_x, dir_sign, gid, start_idx):
        cols = max(1, int((W - radius * 3) // spacing) + 1)
        cols = min(cols, max(count, 1))
        idx = 0
        row = 0
        while idx < count:
            n_this_row = min(cols, count - idx)
            row_width = (n_this_row - 1) * spacing
            y_start = W / 2.0 - row_width / 2.0
            for c in range(n_this_row):
                i = start_idx + idx
                group_id[i] = gid
                direction[i] = dir_sign
                y0[i] = np.clip(y_start + c * spacing + rng.normal(0, 0.05),
                                 radius * 1.5, W - radius * 1.5)
                x0[i] = base_x + dir_sign * row * spacing + rng.normal(0, 0.05)
                idx += 1
            row += 1

    if nA > 0:
        place_block(nA, 0.0, 1.0, 0, 0)
    if nB > 0:
        place_block(nB, L, -1.0, 1, nA)

    pos = np.stack([x0, y0], axis=1)
    e_goal = np.stack([direction, np.zeros(n_peds)], axis=1)
    goal_x = np.where(direction > 0, L + 2.0, -2.0)

    def reached_fn(pos_t):
        return (direction > 0) & (pos_t[:, 0] >= goal_x) | (direction < 0) & (pos_t[:, 0] <= goal_x)

    def wall_fn(pos_t, rad):
        return _corridor_wall_repulsion(pos_t, rad, W, cfg["A_wall"], cfg["B_wall"])

    return pos, e_goal, None, reached_fn, wall_fn, False, group_id


_SETUP = {
    "corridor": _setup_corridor,
    "corridor_soft": _setup_corridor,   # stessa geometria, cambiano solo A/B in cfg
    "corridor_groups": _setup_corridor_groups,   # gruppi coesi, §13 report
    "corridor_two_groups": _setup_corridor_two_groups,   # scontro frontale a 2 gruppi, §15
    "cross": _setup_cross,
    "open_field": _setup_open_field,
}


def simulate_run(n_peds, cfg, seed, scenario="corridor"):
    """Simula un run SFM in uno degli scenari (vedi docstring modulo).

    Ritorna dict ped_id(int) -> lista di (t_idx, x, y) campionata a dt_out
    (solo istanti in cui il pedone e' ancora "in scena": non ha raggiunto
    l'uscita/obiettivo e non ha superato la duration). t_idx e' un indice
    intero di frame (passo=1 <=> dt_out), coerente con la convenzione
    frame_idx usata nella pipeline ACME (data/load_raw.py).
    """
    rng = np.random.default_rng(seed)
    tau0 = cfg["tau"]
    A, B = cfg["A"], cfg["B"]
    radius = cfg["radius"]
    dt_sim, dt_out = cfg["dt_sim"], cfg["dt_out"]
    duration = cfg["duration"]

    N = n_peds
    setup_out = _SETUP[scenario](N, cfg, rng)
    if len(setup_out) == 7:
        pos, e_goal_fixed, goal_point, reached_fn, wall_fn, dynamic_goal, group_id = setup_out
    else:
        pos, e_goal_fixed, goal_point, reached_fn, wall_fn, dynamic_goal = setup_out
        group_id = None
    v0_i = np.clip(rng.normal(cfg["v0_mean"], cfg["v0_std"], N), 0.5, 2.2)
    if group_id is not None:
        # v0 sincronizzata nel gruppo (stessa velocita' desiderata per tutti
        # i membri, presa dal primo estratto): "traiettorie uguali" richiede
        # non solo stessa direzione/corsia ma anche stessa andatura.
        for g in np.unique(group_id):
            idx = np.where(group_id == g)[0]
            v0_i[idx] = v0_i[idx[0]]
    tau = np.full(N, tau0)
    rad = np.full(N, radius)

    if dynamic_goal:
        diff = goal_point - pos
        e_goal = diff / (np.linalg.norm(diff, axis=1, keepdims=True) + 1e-9)
    else:
        e_goal = e_goal_fixed
    vel = e_goal * v0_i[:, None] * 0.5

    n_steps = int(duration / dt_sim)
    out_every = max(1, int(round(dt_out / dt_sim)))
    alive = np.ones(N, dtype=bool)

    traj = {i: [] for i in range(N)}
    for step in range(n_steps):
        if dynamic_goal:
            diff = goal_point - pos
            e_goal = diff / (np.linalg.norm(diff, axis=1, keepdims=True) + 1e-9)
        f0 = _driving_force(vel, e_goal, v0_i, tau)
        frep = _pairwise_repulsion(pos, rad, A, B)
        fwall = wall_fn(pos, rad)
        fcoh = _group_cohesion_force(pos, group_id, cfg.get("group_cohesion_k", 0.8),
                                      cfg.get("group_cohesion_max_force", 2.0)) \
            if group_id is not None else 0.0
        acc = f0 + frep + fwall + fcoh
        vel = vel + acc * dt_sim
        speed = np.linalg.norm(vel, axis=1)
        too_fast = speed > 2.5
        if too_fast.any():
            vel[too_fast] *= (2.5 / speed[too_fast])[:, None]
        pos = pos + vel * dt_sim

        alive &= ~reached_fn(pos)

        if step % out_every == 0:
            t_idx = step // out_every
            for i in range(N):
                if alive[i]:
                    traj[i].append((t_idx, float(pos[i, 0]), float(pos[i, 1])))
        if not alive.any():
            break

    traj = {i: seq for i, seq in traj.items() if len(seq) >= 2}
    return traj


def simulate_open_field_continuous(n_peds, cfg, seed, total_out_steps):
    """Variante "a popolazione continua" di open_field, usata SOLO per la demo
    animata online (§12 report): stesso motore fisico di _setup_open_field,
    ma quando un pedone raggiunge il proprio obiettivo non esce di scena —
    viene subito rigenerato altrove con posizione/obiettivo casuali nuovi,
    sotto un NUOVO id virtuale (trattato come un pedone indipendente ai fini
    del windowing, esattamente come in generate_dataset.py). Cosi' la scena
    resta sempre popolata per tutta la durata della demo, invece di svuotarsi
    dopo i ~20s di un run "a termine" come simulate_run().

    Ritorna traj: virtual_id(int) -> [(t_idx,x,y), ...] — STESSO formato di
    simulate_run(), quindi build_windows_from_run() si applica invariato.
    """
    rng = np.random.default_rng(seed)
    S = cfg["field_size"]
    radius = cfg["radius"]
    tau0 = cfg["tau"]
    A, B = cfg["A"], cfg["B"]
    dt_sim, dt_out = cfg["dt_sim"], cfg["dt_out"]
    N = n_peds

    def spawn_one(i, pos, goal):
        pos[i] = rng.uniform(radius * 1.5, S - radius * 1.5, size=2)
        while True:
            g = rng.uniform(radius * 1.5, S - radius * 1.5, size=2)
            if np.linalg.norm(g - pos[i]) >= S * 0.5:
                goal[i] = g
                return

    pos = np.zeros((N, 2))
    goal = np.zeros((N, 2))
    for i in range(N):
        spawn_one(i, pos, goal)
    v0_i = np.clip(rng.normal(cfg["v0_mean"], cfg["v0_std"], N), 0.5, 2.2)
    tau = np.full(N, tau0)
    rad = np.full(N, radius)
    diff = goal - pos
    vel = (diff / (np.linalg.norm(diff, axis=1, keepdims=True) + 1e-9)) * v0_i[:, None] * 0.5

    virtual_id = np.arange(N)
    next_id = N
    traj = {i: [] for i in range(N)}

    out_every = max(1, int(round(dt_out / dt_sim)))
    step = 0
    out_count = 0
    while out_count < total_out_steps:
        diff = goal - pos
        e_goal = diff / (np.linalg.norm(diff, axis=1, keepdims=True) + 1e-9)
        f0 = _driving_force(vel, e_goal, v0_i, tau)
        frep = _pairwise_repulsion(pos, rad, A, B)
        fwall = _box_wall_repulsion(pos, rad, 0.0, S, 0.0, S, cfg["A_wall"], cfg["B_wall"])
        acc = f0 + frep + fwall
        vel = vel + acc * dt_sim
        speed = np.linalg.norm(vel, axis=1)
        too_fast = speed > 2.5
        if too_fast.any():
            vel[too_fast] *= (2.5 / speed[too_fast])[:, None]
        pos = pos + vel * dt_sim

        reached = np.linalg.norm(pos - goal, axis=1) <= 0.4

        if step % out_every == 0:
            for i in range(N):
                traj[virtual_id[i]].append((out_count, float(pos[i, 0]), float(pos[i, 1])))
            out_count += 1

        for i in np.where(reached)[0]:
            traj[next_id] = []
            virtual_id[i] = next_id
            next_id += 1
            spawn_one(i, pos, goal)
            vel[i] = (goal[i] - pos[i])
            vel[i] = vel[i] / (np.linalg.norm(vel[i]) + 1e-9) * v0_i[i] * 0.5
        step += 1

    traj = {i: seq for i, seq in traj.items() if len(seq) >= 2}
    return traj, out_every
