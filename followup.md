🟩 followup.md — Exécution
Étapes pour Codex

[x] Lire intégralement zemosaic_filter_gui.py depuis le repo.

[x] Repérer la zone où _merge_small_groups est appelée.

[x] Ajouter une nouvelle fonction :
def _apply_hard_merge(groups, settings, logger): ...

[x] Implémenter les règles précisées dans agent.md.

[x] Appeler _apply_hard_merge juste après le preplan coverage-first et avant
la sauvegarde dans overrides_state.preplan_master_groups.

[x] Ajouter le logging dédié.

[x] Ajouter un paramètre merge_threshold = 10 dans les settings si nécessaire.

[x] Exécuter une passe complète de vérification statique.

À tester avec les datasets fournis dans repo

1 dataset fortement recouvrant (Seestar)

1 dataset éclaté en blocs
