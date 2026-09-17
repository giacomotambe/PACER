"""Aggiornamento online di q_tupac(k) in un thread separato dal ciclo di
controllo NMPC (idea discussa esplicitamente con l'utente dopo aver
misurato il costo dell'aggiornamento sincrono: ~0.2ms per ricalcolo del
quantile con pool ~12-14k elementi, O(s) nella dimensione del pool — oggi
piccolo rispetto al ciclo di controllo di 400ms, ma destinato a crescere
senza limite in una missione lunga perche' il pool non ha un cap/finestra
[limite gia' dichiarato nel report]). Disaccoppiare l'aggiornamento dal
thread di controllo rende l'architettura corretta anche quando quel costo
smette di essere trascurabile, invece di scommettere che resti sempre
piccolo.

Divisione del lavoro:
  - thread principale (loop di controllo): ad ogni ciclo risolve la
    maturazione (chi e' vivo, calcolo dello score realizzato via
    mahalanobis_score) — richiede lo stato live corrente, quindi DEVE
    restare sul thread principale — e sottomette (k, score) al worker;
    legge SEMPRE l'ultimo q_tupac(k) disponibile (eventualmente di qualche
    ciclo "vecchio", mai bloccante).
  - questo worker: possiede in esclusiva pool_sorted/s/q_online, fa
    bisect.insort + ricalcolo del quantile (KL-Bernoulli, Teorema 2.3) in
    coda, senza mai bloccare il thread di controllo.

Nessuna magia: la staleness (quanti secondi passano tra la sottomissione di
un evento e il suo effettivo utilizzo nel quantile) e' misurata e riportata
esplicitamente (staleness_s), cosi' come il backlog massimo della coda — se
il worker restasse indietro sistematicamente sarebbe un problema reale, non
va assunto che non succeda.
"""
import threading
import queue
import time
import bisect
import numpy as np

from .quantile_tupac import quantile_from_sorted_pool   # noqa: E402


class OnlineCalibrationWorker(threading.Thread):
    def __init__(self, pool_sorted, s, q_online, alpha_win, delta, h):
        super().__init__(daemon=True, name="tupac-online-calibration")
        self.pool_sorted = pool_sorted   # posseduto in esclusiva da questo thread
        self.s = s                          # idem
        self.q_online = dict(q_online)   # copia iniziale (warm start), poi aggiornata qui
        self.alpha_win, self.delta, self.h = alpha_win, delta, h
        self._q = queue.Queue()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.n_processed = 0
        self.max_backlog = 0
        self.staleness_s = []

    def submit(self, k, r):
        """Chiamato dal thread principale: non blocca mai (coda illimitata,
        push O(1))."""
        self._q.put((k, r, time.perf_counter()))

    def run(self):
        while True:
            try:
                k, r, t_enq = self._q.get(timeout=0.05)
            except queue.Empty:
                if self._stop_event.is_set():
                    return
                continue
            self.max_backlog = max(self.max_backlog, self._q.qsize() + 1)
            bisect.insort(self.pool_sorted[k], r)
            self.s[k] += 1
            q_new = quantile_from_sorted_pool(np.array(self.pool_sorted[k]), self.s[k],
                                                self.alpha_win, self.delta, self.h,
                                                t0=0, bound="kl")
            with self._lock:
                self.q_online[k] = q_new
            self.staleness_s.append(time.perf_counter() - t_enq)
            self.n_processed += 1
            self._q.task_done()

    def snapshot_q(self, N_MPC):
        """Letta dal thread principale ad ogni ciclo: sempre l'ultimo valore
        disponibile, mai un'attesa sul worker."""
        with self._lock:
            return {k: self.q_online[k] for k in range(1, N_MPC + 1)}

    def stop_and_join(self, timeout=30.0):
        """Segnala stop e ASPETTA che la coda si svuoti (flush) prima di
        uscire: il q_tupac_final riportato deve riflettere TUTTI gli eventi
        sottomessi, non un valore troncato a caso a fine missione."""
        self._q.join()
        self._stop_event.set()
        self.join(timeout=timeout)
