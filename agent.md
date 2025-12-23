# Mission: Exposer "Phase 5 GPU chunk (Reproject)" dans l'onglet System (Auto + Spinbox), appliquer uniquement à global_reproject (no-refactor)

## Contexte
Phase 5 "Reproject & Coadd" peut être sous-performante sur certaines configs (micro-chunking).
Nous voulons permettre aux power users de forcer la taille de chunk GPU pour cette phase seulement.

## Objectif UX
Dans l'onglet "System" (GUI Qt):
- Checkbox: "Auto (recommended)"
- Spinbox: "Phase 5 chunk (MB)" (désactivé si Auto coché)
- Side note / tooltip discret:
  "⚠ May cause instability on some laptops / hybrid GPUs. Use with caution."
- Valeurs raisonnables:
  - min 64 MB, max 1024 MB, step 64 MB, default 128 MB (si override utilisé)
- Par défaut: Auto activé.

## Objectif technique
- Stocker dans la config:
  - `phase5_chunk_auto: bool`
  - `phase5_chunk_mb: int`
- Appliquer uniquement lors de la construction du plan de Phase 5, operation="global_reproject":
  - Si Auto: ne rien changer (laisser safety/autotune faire)
  - Si override: forcer `plan.gpu_max_chunk_bytes = phase5_chunk_mb * 1024 * 1024`
    (clamp interne optionnel si champ existe)
  - Ajouter un log INFO: "Phase5 GPU chunk override: XXX MB (global_reproject)"

## Contraintes
- No refactor
- Ne pas impacter les autres phases/ops GPU
- Ne pas changer le comportement batch size=0 vs >1
- Respecter i18n si le projet le fait déjà (traductions FR/EN via util existante)

## Fichiers à modifier (probables)
- `zemosaic_gui_qt.py` (UI: onglet System)
- `zemosaic_config.py` (sauvegarde/chargement des nouveaux champs)
- `zemosaic_worker.py` (application override sur le plan global_reproject)
- éventuellement `zemosaic_localization.py` si les strings passent par un système de traduction

## Acceptation
- Le GUI affiche Auto + spinbox, spinbox grisé quand Auto est actif.
- La config persiste correctement (redémarrage app -> valeur retrouvée).
- En run, si override: log indique la valeur et le plan utilise ce chunk (bytes).
- Si Auto: aucun changement.
