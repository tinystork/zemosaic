# Follow-up — Étape 2: Validation tile_weight dans GPU coadd

## 1) Vérifs rapides dans les logs (run réel)
Lancer un run identique à ton cas "3 master tiles très déséquilibrées".

Attendus dans le log:
- Un bloc DEBUG "tile_weights summary" en Phase 5:
  - min/median/max + ratio
- Un bloc DEBUG côté GPU "gpu_coadd: tile i uses weight ..."
- Les lignes Phase 5 ne doivent plus indiquer `weights_source=alpha_weight2d*tile_weight`
  (ça doit devenir `weights_source=alpha_weight2d` ou similaire),
  puisque tile_weight passe via `tile_weights=`.

Si un WARN "double weighting probable" apparaît:
- C’est qu’on envoie encore des input_weights déjà scalés + tile_weights séparés -> à corriger.

## 2) Tests synthétiques CPU vs GPU
Exécuter:
- `python -m pytest -q` si pytest dispo
ou
- `python tests/test_tile_weight_gpu_coadd.py`

Attendu:
- Test mean: overlap dominé par la tuile la plus pondérée
- Test winsorized: même dominance
- GPU ~ CPU (tolérance float32)

## 3) Contrôle visuel (cas astro)
Sur une mosaïque où une tuile est très profonde:
- En zone de recouvrement, la texture/bruit doit ressembler majoritairement à la tuile profonde
- Les tuiles faibles ne doivent plus “salir” l’overlap (elles ne doivent contribuer que là où la tuile profonde manque)

## 4) Si le problème persiste malgré des poids OK
Alors ce n’est plus le coadd: regarder en priorité l’inter-tile photometric match:
- gains aberrants + clamp (ex: gain ~ 1e-5) => fit instable / mauvais overlap / stats trop fragiles.
- Action suivante: sécuriser le solve photométrique (robust stats / contraintes / fallback offset-only).
