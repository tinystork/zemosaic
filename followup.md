# Follow-up – ASTAP concurrency cap (cpu_count - 2 rule)

Merci d’avoir implémenté la première passe 🙏  
Voici la checklist de vérification et d’éventuels ajustements.

## ✅ Checklist de review

- [ ] Le helper `compute_astap_recommended_max_instances(...)` est bien présent dans `zemosaic_astrometry.py`, documenté, et sans dépendances inutiles.
- [ ] Le helper gère proprement les cas edge (cpu_count=None, exceptions) et retourne toujours `>= 1`.
- [ ] Le helper applique bien la règle : `recommended = min(max(1, cpu - 2), 32)`.

### GUI Qt principal

- [ ] Le `QSpinBox` `astap_max_instances` utilise maintenant `maximum=compute_astap_recommended_max_instances()`, avec un fallback cohérent en cas d’erreur.
- [ ] `_resolve_astap_max_instances()` clamp la valeur de config entre 1 et la limite recommandée.
- [ ] `_apply_astap_concurrency_setting()` utilise toujours `_resolve_astap_max_instances()` et met à jour :
  - [ ] `os.environ["ZEMOSAIC_ASTAP_MAX_PROCS"]`
  - [ ] `set_astap_max_concurrent_instances(...)` (si disponible)
- [ ] Si une ancienne config contient une valeur > limite recommandée, le spinbox affiche bien la valeur clampée après chargement du GUI.

### Filter GUI Qt

- [ ] `zemosaic_filter_gui_qt.py` importe `compute_astap_recommended_max_instances` (avec garde `try/except` si nécessaire).
- [ ] `_populate_astap_instances_combo()` utilise le helper pour calculer `cap`, avec fallback sur l’ancien comportement (`cpu_count // 2`) en cas d’erreur.
- [ ] La combo “Max ASTAP instances” propose la plage `[1 .. min(os.cpu_count() - 2, 32)]`.
- [ ] Le warning multi-instance (popup “Access violation” / “ASTAP Concurrency Warning”) fonctionne toujours dès que l’utilisateur choisit `> 1`.

### Config & compat

- [ ] `DEFAULT_CONFIG["astap_max_instances"]` est toujours défini et cohérent (1 ou autre valeur raisonnable).
- [ ] `get_astap_max_instances()` renvoie une valeur `>= 1` et reste compatible avec le reste du code.
- [ ] Aucun changement n’a été apporté aux pipelines CPU/GPU de stacking / mosaïque.

## 🧪 Tests manuels à effectuer

1. **Machine avec peu de threads (ex: 4 ou 8 threads)**  
   - [ ] Vérifier que la limite GUI = `min(cpu_count - 2, 32)` (ex: 8 threads → max 6).
   - [ ] Lancer un run et vérifier dans les logs que la valeur passée à ASTAP correspond bien au réglage choisi (clampé).
2. **Machine avec beaucoup de threads (ex: 32 ou 64 threads)**  
   - [ ] Vérifier que la limite GUI n’excède jamais 32.
3. **Ancienne config qui contenait une valeur élevée**  
   - [ ] Modifier manuellement `zemosaic_config.json` pour mettre `astap_max_instances` à une valeur absurde (ex: 80).
   - [ ] Relancer le GUI QT :
     - [ ] Le spinbox doit afficher une valeur `<= min(cpu_count - 2, 32)`.
     - [ ] La valeur runtime appliquée à ASTAP doit être identique à celle affichée.

Si tout passe cette checklist, on considérera la tâche comme **terminée et stable** pour les utilisateurs “lambda”, tout en gardant la possibilité de tweaker finement via la config/env pour les power users.
````
