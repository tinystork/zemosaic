# followup.md

## Ce que tu dois livrer
- Un patch Git prêt à merger, limité à :
  - `zemosaic_worker.py`
  - `zemosaic_align_stack.py`

- Des logs plus explicites (niveau DEBUG_DETAIL) montrant :
  - le mode (EQ / ALT_AZ) détecté par groupe en Phase 3
  - la décision “global_enabled vs tile_enabled” pour ALT-AZ cleanup

## Checklist de review (avant de rendre)
### 1) Vérification “pas de régression SDS / Grid”
- [ ] `grid_mode.py` : inchangé
- [ ] Aucune modification de logique SDS (chercher `sds_` dans diff : uniquement des impacts indirects acceptables)
- [ ] Le comportement global hors Phase 3 n’est pas altéré inutilement

### 2) Worker : gating ALT-AZ cleanup par groupe
- [ ] Ajout d’un helper `_infer_group_eqmode(...)` (aucune lecture disque)
- [ ] Si `contains_altaz=False` alors `altaz_cleanup_effective_flag` forcé à False (WARN si l’option était demandée)
- [ ] En Phase 3, `create_master_tile(... altaz_cleanup_enabled=altaz_cleanup_for_tile ...)` :
  - [ ] ALT_AZ => True
  - [ ] EQ => False
- [ ] Logs DEBUG_DETAIL clairs (1 ligne / tile max, pas de spam)

### 3) Align stack : CPU kappa fallback chunké
- [ ] `_cpu_stack_kappa_fallback` ne fait plus `np.stack(all_frames)` sur tout H
- [ ] Chemin “fast” conservé si petite taille (pas de ralentissement sur petits stacks)
- [ ] Mode chunk :
  - [ ] mémoire bornée (pas de temp géant)
  - [ ] support RGB (N,H,W,3) et mono (N,H,W)
  - [ ] support weights 1D au minimum (le plus courant)
- [ ] `rejected_pct` cohérent avec l’ancien fallback

### 4) Self-tests
- [ ] `_selftest_cpu_kappa_chunk_equivalence()` existe, non exécuté par défaut
- [ ] `_selftest_infer_group_eqmode()` existe, non exécuté par défaut

## Scénarios de validation manuelle (rapides)
### A) “EQ-only”
- Simuler `contains_altaz=False`, option ALT-AZ cleanup cochée
- Attendu :
  - WARN “requested but no ALT_AZ frames detected -> disabled”
  - Phase 3 : `tile_enabled=False` partout
  - pas de master tiles vides

### B) “Mix EQ + ALT_AZ”
- Groupe EQ (entry["_eqmode_mode"]="EQ") + groupe ALT_AZ (entry["_eqmode_mode"]="ALT_AZ")
- Attendu :
  - EQ : `tile_enabled=False`
  - ALT_AZ : `tile_enabled=True`

### C) “Big group fallback CPU”
- Forcer un cas (N élevé) qui déclenche chunking (ou baisser le seuil pour test)
- Attendu :
  - pas d’alloc > quelques centaines de MB
  - pas de crash nanmedian
  - sortie float32 OK

## Format de sortie
- Donne le diff/patch final
- Donne un court résumé (5-10 lignes) des changements
- Donne les points de risque éventuels (s’il y en a) + pourquoi tu estimes que SDS/Grid sont safe
