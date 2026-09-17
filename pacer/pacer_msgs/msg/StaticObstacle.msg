# Ostacolo STATICO estratto dal local costmap di Nav2 (celle "lethal"/
# inflated clusterizzate, vedi pacer_nav2_controller) — trattato come
# costo soft nell'NMPC (mai come vincolo rigido, riservato ai pedoni:
# vedi nota vault "Sim2Real - dettagli implementativi", §2.3).
geometry_msgs/Point xy
float64 d_safe
