# Mission — Mettre le mode SDS en conformité avec le workflow “ZeSupaDupStack”

## Rôle de cette mission

Tu es Codex High.  
Ta mission est de faire en sorte que le **mode SDS** de ZeMosaic suive exactement le workflow conceptuel décrit ci-dessous, **sans modifier le comportement du mode normal (SDS OFF)**.

Tu dois considérer ce document comme la **spécification fonctionnelle** du mode SDS.

---

## Champ d’action (IMPORTANT)

### Fichiers que tu peux modifier (code) :

- `zemosaic_worker.py`  
  - uniquement les parties qui concernent :
    - la détection du mode SDS,
    - le pipeline SDS (création de lots, méga-tuiles, renormalisation inter-méga-tiles, stack final),
    - la finalisation SDS.

### Fichiers que tu peux éventuellement toucher **légèrement** :

- `zemosaic_config.py`  
  - **uniquement** pour ajouter/ajuster des paramètres spécifiques SDS si nécessaire (sans changer les clés existantes ni les valeurs par défaut globales hors SDS).
- `en.json`, `fr.json`  
  - pour ajouter des **nouvelles chaînes de log SDS** si tu en as réellement besoin,
  - sans supprimer ni renommer des clés existantes.

### Fichiers que tu ne dois PAS modifier :

- `zemosaic_align_stack.py`
- `lecropper.py`
- `zemosaic_gui.py`
- `zemosaic_gui_qt.py`
- `zemosaic_filter_gui.py`
- `zemosaic_filter_gui_qt.py`
- tout autre fichier non listé ci-dessus.

### Invariant fondamental

> Le pipeline **non-SDS (SDS OFF)** doit conserver **strictement le même comportement** qu’avant tes modifications.

- Même séquence de phases (Master Tiles, Phase 4/4.5, Phase 5).
- Même logique de normalisation/empilement pour les Master Tiles.
- Même comportement pour `batch_size=0` et `batch_size>1`.
- Même usage GPU/CPU, memmap, etc.

Tu peux facturer un peu de code générique **si cela ne change pas la sortie du mode normal**, mais **tu ne dois jamais modifier la structure générale des phases non-SDS**.

---

## Workflow SDS (ZeSupaDupStack) — SPÉCIFICATION

Le mode SDS n’est **pas** un mode mosaïque classique.  
C’est un mode **super-stack par lots**, conçu pour :

- découper intelligemment les données du Seestar,
- normaliser *localement*,
- produire des **méga-tuiles photométriquement propres**,
- puis les empiler dans **UN SEUL grand stack global**.

Tu dois t’assurer que le code SDS reflète ce workflow de bout en bout.

### 1️⃣ Entrée : le filtre SDS prépare les lots

Le filtre (GUI Qt) :

- analyse les positions WCS des images,
- estime la coverage,
- applique une taille de lot cible,
- tient compte des zones de recouvrement.

Il produit une liste de **LOTS SDS** de ce type :

- Lot 1 : images {1,2,3,…}  
- Lot 2 : images {…}  
- etc.

**Important :**  
En mode SDS, ces **lots SDS remplacent complètement les Master Tiles**.  
Il ne faut plus reconstruire de Master Tiles dans le pipeline SDS.

---

### 2️⃣ Pour chaque lot SDS → création d’une méga-tuile

Pour chaque lot, le worker doit :

1. **Charger les images du lot**
   - débayer,
   - corriger les pixels chauds,
   - corriger les gradients locaux si nécessaire,
   - appliquer les mêmes filtres de qualité de base que dans le pipeline classique.

2. **Aligner les images du lot directement sur le WCS global**
   - pas de WCS intermédiaire “local”,
   - alignement dans la géométrie finale de la mosaïque.

3. **Normalisation INTRA-lot**
   - mettre les images du lot au même niveau photométrique,
   - harmoniser le fond de ciel (match background),
   - appliquer la même logique de robustification que pour les Master Tiles :
     - winsor / kappa, rejet des valeurs extrêmes,
     - rejet des images floues/ratées si la config l’indique.

   ➜ Objectif : **une tuile par lot déjà photométriquement propre.**

4. **Coadd du lot → MEGA-TILE**
   - effectuer un coadd complet (GPU si disponible),
   - produire :
     - une image `H×W×3`,
     - une carte `coverage H×W`.

Le résultat est une **méga-tuile SDS** qui représente l’ensemble du lot comme une seule “super image”.

---

### 3️⃣ Après tous les lots → normalisation INTER-méga-tiles

Une fois tous les lots traités, on a :

- `MT0, MT1, MT2, ..., MTn` (méga-tiles)  
- chacune avec une coverage associée.

Chaque méga-tuile peut avoir un niveau de flux/fond légèrement différent.

Tu dois appliquer :

1. **Choix d’une méga-tuile de référence**
   - critère possible :
     - meilleure coverage,
     - plus centrale,
     - ou première, si la config le précise.
   - cette tuile ancre le flux global (son gain = 1.0).

2. **Calcul des médianes sur les zones couvertes**
   - pour chaque méga-tuile :
     - ne considérer que les pixels où `coverage > 0`,
     - calculer une médiane robuste (en ignorant NaN et valeurs aberrantes).

3. **Renormalisation globale**
   - pour chaque méga-tuile (sauf la référence) :
     - calculer `gain_i = median_ref / median_i`,
     - appliquer ce gain à toute la méga-tuile.

➜ Après cette étape, **toutes les méga-tiles doivent être sur la même échelle photométrique**.

---

### 4️⃣ Empilement global des méga-tiles

Une fois toutes les méga-tiles normalisées globalement :

1. **Alignement des méga-tiles dans le WCS final**
   - elles sont déjà dans le bon WCS,
   - donc :
     - pas de reprojection lourde,
     - tout au plus recomposition/cropping léger si besoin.

2. **Coadd global**
   - empiler toutes les méga-tiles pour produire :
     - `sds_mosaic_data_HWC`,
     - `sds_coverage_HW`,
     - éventuellement `alpha_map`.

**Important :**  
À ce stade, **il ne faut plus appliquer de “match background”**.  
La normalisation globale inter-méga-tiles a déjà aligné les niveaux.

---

### 5️⃣ Finalisation SDS (polish final)

Sur `sds_mosaic_data_HWC` + `sds_coverage_HW` (+ `alpha_map`), appliquer :

1. **Coverage cut**
   - masque des pixels avec coverage trop faible (coverage normalisée < seuil),
   - convertir ces pixels en NaN + coverage=0.

2. **Two-pass coverage renorm (optionnel)**
   - homogénéiser les bordures si l’option est activée.

3. **Quality crop**
   - utiliser `lecropper` ou ZeQualityMT pour :
     - éliminer les zones instables,
     - rogner les bords dégradés.

4. **Alt-Az cleanup**
   - suppression des zones impactées par rotation de champ ou dérive alt-az.

5. **Autocrop WCS**
   - cropping final basé sur la coverage valide,
   - mise à jour du WCS global pour refléter le crop.

---

### 6️⃣ Sauvegarde finale

Le pipeline SDS doit ensuite :

- écrire le FITS final :
  - image principale,
  - extension(s) de coverage,
  - WCS final ajusté (cropped),
- générer PNG/JPG si demandé,
- enregistrer :
  - des métriques de qualité,
  - des logs sur :
    - nombre de lots,
    - temps par lot,
    - temps total SDS,
    - index de la méga-tuile de référence et gains appliqués.

---

## Résumé SDS ultra-court

> `SDS = lots → méga-tiles → renorm globale → stack final → polish → save`

Tu dois adapter le code SDS existant pour qu’il respecte ce flux,  
**sans altérer le pipeline non-SDS**.
