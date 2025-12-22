# agent.md — Fix trou central en mode "existing master tiles" via alpha_from_coverage (Option B)

## Contexte / Symptôme
En mode "I'm using master tiles (skip clustering_master tile creation)", l'image finale présente parfois un énorme trou (zone nan/opaque) alors que la carte de coverage montre un recouvrement correct.

Le log indique :
- phase6: alpha_union received -> alpha_final propagé
- puis: nanized X pixels where coverage/alpha == 0

Hypothèse : dans ce mode seulement, l’alpha reprojeté (alpha_union) peut être incohérent (ex: 0 dans des zones couvertes), et comme on nanize en fonction de l’alpha, on “perce” un trou artificiel.

## Objectif (Option B)
Quand on est en mode "use_existing_master_tiles", dériver l’ALPHA FINAL à partir du coverage final :
- alpha_final[u,v] = 255 si coverage[u,v] > 0
- alpha_final[u,v] = 0 sinon

Et ne pas laisser un alpha_union incohérent dégrader le rendu final.

## Contraintes
- Patch chirurgical.
- Ne pas changer le comportement normal hors de ce mode.
- Ne pas refactorer l’architecture.
- Zéro impact sur batch size, GPU/CPU, etc.

## Fichiers à modifier
- `zemosaic_worker.py` uniquement (idéalement)
  - `run_hierarchical_mosaic_classic_legacy(...)`
  - `assemble_final_mosaic_reproject_coadd(...)`

## Plan d’implémentation

- [x] Propager un bool "existing_master_tiles_mode" vers l’assembleur final
  - Ajouter un paramètre optionnel à `assemble_final_mosaic_reproject_coadd` :
    - `existing_master_tiles_mode: bool = False`
  - Dans `run_hierarchical_mosaic_classic_legacy`, lors de l’appel à `assemble_final_mosaic_reproject_coadd`, passer :
    - `existing_master_tiles_mode=use_existing_master_tiles_config` (ou l’équivalent déjà présent)

⚠️ Important : garder une valeur par défaut pour ne rien casser ailleurs.

- [x] Rebuild alpha_final depuis coverage en mode existing master tiles
  - Dans `assemble_final_mosaic_reproject_coadd`, à l’endroit où `alpha_union` est converti en `alpha_final` (phase6), insérer juste après l’obtention de `alpha_final` et avant toute logique “coverage/alpha == 0” :

    Pseudo-code :

    ```python
    if existing_master_tiles_mode and coverage is not None:
        cov_mask = coverage > 0
        if alpha_final is None:
            alpha_final = (cov_mask.astype(np.uint8) * 255)
            log("alpha_from_coverage: alpha_final was None -> rebuilt")
        else:
            # détecter mismatch : coverage>0 mais alpha==0
            alpha0 = (alpha_final == 0)
            mismatch = np.count_nonzero(cov_mask & alpha0)
            if mismatch > 0:
                total_cov = np.count_nonzero(cov_mask)
                pct = (100.0 * mismatch / max(1, total_cov))
                log(f"alpha_from_coverage: overriding alpha_final (mismatch={mismatch} px, {pct:.2f}% of covered)")
                alpha_final = (cov_mask.astype(np.uint8) * 255)
    ````

    * Forcer dtype `uint8`, shape identique au coverage.
    * Conserver `alpha_union` tel quel pour les autres modes.

- [x] Logging / Observabilité

  - Ajouter un log INFO (et si tu veux un callback GUI type `[INFO] ...`) quand on override :

  * nb pixels mismatch
  * % des pixels couverts impactés
  * mention claire “existing_master_tiles_mode”

- [ ] Critères d’acceptation

  * En mode existing master tiles, plus de trou central si le coverage indique une couverture.
  * Les stats “nanized pixels where coverage/alpha == 0” ne doivent plus nanizer des pixels `coverage>0` à cause de l’alpha.
  * Hors de ce mode : comportement inchangé.

## Notes

* Ce fix est volontairement conservateur : en mode existing master tiles, on privilégie la vérité “coverage” (résultat de l’assemblage) plutôt qu’un alpha potentiellement corrompu/inversé venant des fichiers en entrée.
