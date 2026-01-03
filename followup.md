# followup.md — Notes d’implémentation & plan de QA (V1 Resume)

## Rappels importants (anti-régression)
- Default = `resume="off"` → **ne rien changer** au comportement actuel.
- La reprise V1 ne doit s’activer que dans `run_hierarchical_mosaic_classic_legacy()` et uniquement si on est bien dans un contexte classic legacy.
- Ne pas toucher SDS / grid / “I’m using master tiles”.
- Ne pas ajouter de dépendances.

## Détails pratiques (sérialisation Astropy)
### Écriture
- `header_str` : utiliser `header_obj_updated.tostring(sep="\n", endcard=False, padding=False)` (si padding non supporté, rester simple).
- Stocker `header_str` dans `phase1_processed_info.json`.

### Lecture
- `header = fits.Header.fromstring(header_str, sep="\n")`
- `wcs = astropy.wcs.WCS(header)`
- Injecter `entry["header"]`, `entry["wcs"]` dans la structure mémoire attendue par Phase 2.

## Validation cache (recommandation)
Quand `resume=auto`, refuser si :
- `cache_manifest.json` absent/invalide
- `phase1.done` absent
- `schema_version != 1`
- `pipeline != "classic_legacy"`
- `run_signature` ne matche pas
- `phase1_processed_info.json` absent / illisible / liste vide
- Un `path_preprocessed_cache` référencé n’existe pas

Quand `resume=force` :
- ignorer seulement le mismatch de signature
- MAIS refuser si fichiers essentiels manquants (manifest/marker/data/cache .npy)

## Plan de QA manuel (à exécuter)
### Cas A — comportement inchangé
1) Retirer toute config `resume` (ou mettre `off`)
2) Lancer un run
3) Vérifier dans les logs que `.zemosaic_img_cache` est supprimé/recréé au début comme avant

### Cas B — reprise auto OK
1) Mettre `resume="auto"`
2) Mettre `cache_retention_mode="keep"` (sinon la reprise ne sert à rien)
3) Lancer un run complet (Phase 1 s’exécute et produit manifest/marker)
4) Relancer immédiatement
5) Vérifier :
   - log “Phase 1 skipped (resume)”
   - Phase 2 démarre correctement
   - pas d’erreur WCS (centres/coords OK)

### Cas C — reprise auto refusée (input modifié)
1) Après un run qui a produit le cache, ajouter un nouveau FITS dans `input_folder`
2) Relancer avec `resume="auto"`
3) Vérifier :
   - reprise refusée + raison “signature mismatch / input changed”
   - cache renommé/supprimé puis run complet normal

### Cas D — force
1) Reprendre Cas C, mais mettre `resume="force"`
2) Vérifier :
   - WARN “force: signature mismatch ignored”
   - reprise acceptée SI et seulement SI les `.npy` référencés existent

## Points à surveiller
- Progress / ETA : si Phase 1 est sautée, avancer `current_global_progress` pour éviter une barre bloquée au début.
- Ne pas crasher si l’écriture du manifest échoue : log WARN et continuer.
- Le manifest ne doit pas être écrit quand `resume="off"` (pour garder le comportement ultra inchangé par défaut).

## Output attendu
- Patch unique sur `zemosaic_worker.py`
- Nouveaux fichiers générés seulement si `resume != off` :
  - `.zemosaic_img_cache/cache_manifest.json`
  - `.zemosaic_img_cache/phase1_processed_info.json`
  - `.zemosaic_img_cache/phase1.done`
````
