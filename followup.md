# followup.md — Vérifs & Validation (alpha_from_coverage en mode existing master tiles)

## 1) Repro / Test manuel rapide (cas M31)
- Utiliser exactement le dataset + config qui déclenche le problème (use_existing_master_tiles=true, quality_crop=false).
- Lancer un run.
- Attendus :
  - La preview PNG n’a plus de “gros trou” central.
  - Le FITS final contient une extension ALPHA (si c’était déjà le cas), mais cette ALPHA doit maintenant correspondre au coverage.

## 2) Check quantitatif simple (log)
Attendre un log du type :
- `alpha_from_coverage: overriding alpha_final (mismatch=..., ..% of covered)`
ou
- `alpha_from_coverage: alpha_final was None -> rebuilt`

Et vérifier qu’il n’y a plus de “nanized pixels” induits par un alpha incohérent.

## 3) Sanity check : cohérence alpha vs coverage
Ajouter temporairement (ou via debug local) un petit calcul :
- `mismatch_after = count((coverage>0) & (alpha_final==0))`
Attendu : `mismatch_after == 0`

## 4) Non-régression (mode normal)
- Lancer un run classique (sans existing master tiles).
- Vérifier :
  - pas de changement de comportement,
  - alpha_union continue d’être propagé tel qu’avant,
  - pas de nouveaux logs alpha_from_coverage.

## 5) Nettoyage
- Garder les logs INFO (utiles).
- Ne pas ajouter de dépendances.
