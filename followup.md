# Follow-up — Vérifications après patch "plugged-aware"

## 1) Vérifier la sémantique batterie dans les logs
- [ ] Un PC avec batterie MAIS sur secteur doit logguer:
      power_plugged=True, on_battery=False, has_battery=True
      et NE PAS inclure une "raison" qui suggère une limitation batterie.
- [ ] Sur batterie (on_battery=True) doit logguer explicitement: on_battery_clamp

## 2) Vérifier la logique de clamp Safe Mode
- [ ] safe_mode=1 ET on_battery=True => budget GPU clamp agressif (ex 128MB max, puis cap VRAM)
- [ ] safe_mode=1 ET hybrid=True ET power_plugged=False => clamp agressif (hybrid_unplugged_clamp)
- [ ] safe_mode=1 ET hybrid=True MAIS power_plugged=True => budget par défaut plus large (ex 256MB) avant cap VRAM

## 3) Vérifier Phase 5 Auto chunk (Reproject)
- [ ] En Auto, un log unique doit apparaître:
      "Phase5 chunk AUTO: applied=...MB (... power_plugged=..., on_battery=..., hybrid=..., reasons=[...])"
- [ ] Ce log doit indiquer clairement:
      - valeur appliquée (MB)
      - cap VRAM si appliqué
      - raison du clamp éventuel

## 4) Non-régression
- [ ] Mode USER chunk (override) doit rester fonctionnel et prioritaire.
- [ ] Les logs existants "Phase5 GPU: bump rows_per_chunk ... (plugged)" doivent rester cohérents.
- [ ] Pas de modification des autres phases / pas de refactor.

## 5) Attendu côté perf
- [ ] Sur secteur, Phase 5 Auto ne doit plus rester figé à 128MB uniquement à cause de hybrid_graphics.
- [ ] Sur batterie, on conserve une posture conservative (sécurité/stabilité).
