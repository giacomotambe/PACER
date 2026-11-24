# Storico posizioni di UN pedone tracciato (cadenza dt_out=0.4s, al massimo
# obs_len campioni, protocollo 8:12). Rev. 2026-11-11: pubblicato anche se
# ancora incompleto (1..obs_len-1 campioni, pedone appena entrato nel volume
# di cattura o dopo un'occlusione), cosi' l'MPC ne tiene conto: il
# predittore usa il modello solo per gli storici completi.
# Prodotto da pacer_qualisys_bridge (oggi: markers Qualisys, id = nome
# rigid body; in futuro: un modulo di riconoscimento/tracking reale con la
# STESSA interfaccia, vedi nota vault "Sim2Real - dettagli implementativi").
string id
geometry_msgs/Point[] history
