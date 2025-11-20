# Follow-up — Checklist pour aligner le code SDS sur le workflow ZeSupaDupStack

Cette checklist te guide pas à pas.  
Tu dois rester concentré sur **le pipeline SDS** et **ne pas modifier le comportement non-SDS**.

---

## Étape 1 — Identifier la bifurcation SDS ON / OFF

1. [x] Dans `zemosaic_worker.py`, localise la fonction principale (souvent `run_hierarchical_mosaic(...)`) qui :
   - [x] construit le plan WCS global,
   - [x] décide si SDS est activé,
   - [x] appelle actuellement `assemble_global_mosaic_sds(...)` ou un équivalent.

2. [x] Vérifie que :
   - [x] quand SDS est **OFF** :
     - [x] le pipeline classique (Master Tiles + Phase 5) reste inchangé,
   - [x] quand SDS est **ON** :
     - [x] **on n’utilise pas** le Mosaic-First global “toutes brutes d’un coup” comme pipeline de stacking,
     - [x] mais un pipeline **par lots** conforme au design SDS.

Si actuellement SDS réutilise `assemble_global_mosaic_first_impl(...)` pour stacker toutes les brutes, tu dois **remplacer cette logique par le pipeline par lots**, tout en conservant éventuellement Mosaic-First uniquement comme aide à la construction du WCS global si nécessaire.

---

## Étape 2 — Utiliser les LOTS SDS

1. [x] Vérifie comment les lots SDS sont transmis au worker :
   - [x] via `preplan_path_groups`,
   - [x] ou via un plan interne SDS.

2. [x] Assure-toi que, en mode SDS :

   - [x] **les lots SDS remplacent la logique “Master Tiles”** :
     - [x] pas de création de Master Tiles,
     - [x] pas de Phase 3 classique dans le chemin SDS.

3. [x] Si certains morceaux de code Master Tiles sont réutilisables (normalisation, stacking), tu peux les appeler, mais **sans relancer la Phase 3 globale**.

---

## Étape 3 — Pour chaque lot SDS → produire une méga-tuile

Pour chaque lot SDS, implémente :

1. [x] Lecture / préparation des brutes :
   - [x] réutiliser les fonctions déjà existantes pour :
     - [x] débayer,
     - [x] corriger les pixels chauds,
     - [x] appliquer les mêmes pré-traitements que pour les Master Tiles.

2. [x] Alignement des images du lot :
   - [x] les aligner directement dans le WCS global de la mosaïque.

3. [x] Normalisation intra-lot :
   - [x] utiliser la même logique de normalisation que pour les Master Tiles :
     - [x] linear fit / noise variance,
     - [x] match background local,
     - [x] rejet sigma / winsor si activé,
     - [x] filtre d’images floues si déjà implémenté.

4. [x] Coadd du lot :
   - [x] produire une **méga-tuile** :
     - [x] image `H×W×3`,
     - [x] coverage `H×W`.

5. [x] Stocker les méga-tiles dans une structure en mémoire (liste) plutôt que dans des répertoires temporaires vides.

Ajoute des logs raisonnables (INFO / INFO_DETAIL) :  
- [x] index du lot,  
- [x] nombre d’images,  
- [x] temps de traitement du lot.

---

## Étape 4 — Normalisation inter-méga-tiles

1. [x] Implémente une fonction ou logique interne pour :

   - [x] choisir une **méga-tuile de référence** :
     - [x] idéalement celle avec le meilleur “poids de coverage”,
     - [x] sinon une stratégie simple (méga-tuile centrale, ou première).

2. [x] Pour chaque méga-tuile :

   - [x] utiliser sa coverage pour ne prendre en compte que les pixels avec `coverage > 0`,
   - [x] calculer une médiane robuste (en ignorant NaN).

3. [x] Calculer un gain par méga-tuile :

   - [x] `gain_ref = 1.0` pour la tuile de référence,
   - [x] pour les autres : `gain_i = median_ref / median_i`,
   - [x] appliquer ce gain à chaque méga-tuile.

Assure-toi que :
- [x] aucun gain n’est NaN ou infini (fallback sur 1.0 si problème),
- [x] les méga-tiles restent dans un type float32 cohérent.

Ajoute un log :
- [x] index de la méga-tuile de référence,
- [x] médiane de référence,
- [x] quelques gains appliqués (au moins min / max / quelques exemples).

---

## Étape 5 — Stack global des méga-tiles

1. [x] Empile toutes les méga-tiles normalisées :

   - [x] elles sont déjà dans le bon WCS,
   - [x] tu peux utiliser un stacker commun (moyenne / winsor / kappa-sigma) pour produire :
     - [x] `sds_mosaic_data_HWC`,
     - [x] `sds_coverage_HW`.

2. [x] **Ne réapplique pas de match background à ce stade.**  
   - [x] la normalisation inter-méga-tiles a déjà aligné les niveaux globaux.

3. [x] Prépare une éventuelle `alpha_map` si nécessaire.

---

## Étape 6 — Finalisation SDS

Réutilise au maximum les briques existantes déjà utilisées pour la Phase 5 classique, mais applique-les sur les sorties SDS :

1. [x] Coverage cut :
   - [x] à partir de `sds_coverage_HW`,
   - [x] remplace les pixels de coverage trop faible par NaN dans `sds_mosaic_data_HWC` + coverage=0.

2. [x] Two-pass coverage renorm (si activé) :
   - [x] applique la même logique que pour le pipeline classique.

3. [x] Quality crop :
   - [x] utilise `lecropper` / ZeQualityMT comme dans la Phase 5 classique,
   - [x] mets à jour la coverage et l’alpha map en conséquence.

4. [x] Alt-Az cleanup :
   - [x] pareil que dans le pipeline classique.

5. [x] Autocrop WCS :
   - [x] applique le cropping basé sur la coverage,
   - [x] ajuste le WCS global pour refléter ce crop.

---

## Étape 7 — Sauvegarde finale et fallbacks

1. [x] Quand SDS réussit :

   - [x] `final_mosaic_data_HWC`, `final_mosaic_coverage_HW`, `final_alpha_map` doivent provenir du pipeline SDS,
   - [x] on ne relance pas de Phase 5 classique sur les Master Tiles.

2. [x] En cas d’échec SDS (erreur, NaN partout, etc.) :

   - [x] loguer clairement le problème,
   - [x] utiliser les fallbacks existants (Mosaic-First, puis Master Tiles),
   - [x] **sans modifier la logique de fallback actuelle**.

---

## Étape 8 — Vérifications finales

Avant de conclure ta modification :

- [x] vérifie que :
  - [x] lorsque SDS est OFF, le code ne prend aucun des nouveaux chemins SDS,
  - [x] les sorties d’un run non-SDS restent identiques ou équivalentes,
  - [x] les chemins `batch_size=0` et `batch_size>1` ne sont pas impactés.

- [x] ne modifie pas :
  - [x] la structure des phases non-SDS,
  - [x] les signatures des fonctions publiques existantes.

Tu peux commenter brièvement les sections clés du nouveau pipeline SDS pour documenter le design “lot → méga-tuile → renorm → stack final → polish → save”.
