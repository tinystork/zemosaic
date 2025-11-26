# Mission Codex High/Max – Retour STRICT au comportement du commit `38c876a` (CPU & GPU)

## 🎯 Objectif

Revenir **strictement** au comportement du commit `38c876a` pour la chaîne classique Master Tile → mosaïque, en CPU **et** en GPU :

- **Même logique, même ordre d’opérations, mêmes poids, mêmes phases**.
- La voie GPU doit produire **la même image** que la voie CPU (à tolérance float près), comme au commit `38c876a`.
- **Aucune “amélioration” ni réinterprétation** : on veut **le code de 38c876a**, réintégré dans la base actuelle, en gardant seulement ce qui est nécessaire pour compiler / fonctionner.

Le dernier run **réellement fonctionnel** (CPU = GPU OK) correspond au commit `38c876a`, log enregistré dans :
- `zemosaic_worker38c876a.log` :contentReference[oaicite:0]{index=0}  

Les images associées montrent :
- Image 3 = **référence bonne** (GPU/CPU sous `38c876a`).
- Image 1 (CPU actuel) = bandes de jointure visibles.
- Image 2 (GPU actuel) = dérive couleur + artefacts.

Le but : **revenir au comportement de l’image 3**, pour CPU et GPU, dans la branche actuelle.

---

## 🗂 Contexte et périmètre

- Repo : **ZeMosaic**
- Branche de travail : **branche actuelle (ex : `V4WIP` ou équivalent)** – ne pas toucher à l’historique git, travailler via modifications de fichiers.
- Commit de référence (golden) : `38c876a`
- Log de référence : `zemosaic_worker38c876a.log` (golden run complet). :contentReference[oaicite:1]{index=1}  

### Fichiers principaux impliqués

À analyser et, si besoin, **ramener exactement à l’état logique de `38c876a`** pour la voie “Master Tiles → Reproject → Coadd” :

- `zemosaic_worker.py` (Phase 3–6, assemblage final, photométrie, normalisation RGB, CPU/GPU, two-pass, etc.) :contentReference[oaicite:2]{index=2}  
- `zemosaic_align_stack.py` (traitement CPU)
- `zemosaic_align_stack_gpu.py` (traitement GPU)
- Éventuellement :
  - `zemosaic_utils.py`
  - `zemosaic_astrometry.py`
  - Toute fonction utilitaire spécifique appelée par la Phase 5 / two-pass / RGB-equalize.

🔒 **Zones à ne pas modifier (sauf nécessité mécanique de compatibilité) :**

- Le mode **SDS / ZeSupaDupStack / Phase 4.5 super-tiles** : **NE PAS TOUCHER LA LOGIQUE SDS**.
- La logique générale du GUI Tk / Qt, sauf pour adapter à des renommages s’il y en a eu.
- Le système de télémétrie / ParallelPlan, sauf si un appel / argument doit être réaligné sur l’API précédente.

---

## 🔍 Symptômes actuels

- En mode **CPU**, l’image finale présente des **bandes de jointure photométriques** (différences de fond / couleur entre tuiles) et des bords noirs résiduels (cf. image 2).
- En mode **GPU**, la mosaïque est largement corrompue (trames verticales massives, cf. image 1), signe que la Phase 5/two-pass GPU diverge fortement du comportement `38c876a`.
- Ces dérives n’étaient **pas présentes** dans le run de référence `38c876a` : le log montre une Phase 5 propre, avec **RGB equalization maîtrisée** et **two-pass coverage renorm GPU** qui fonctionne correctement (cf. image 3). :contentReference[oaicite:3]{index=3}  

### 📑 Constats logs (référence vs runs actuels)

- Golden (`zemosaic_worker38c876a.log`) : intertile auto-tune → 87 paires, photométrie appliquée (`apply_photometric` sur 27 tuiles), two-pass GPU OK (gains non triviaux).
- CPU actuel (`zemosaic_workerCPU.log`) : 87 paires trouvées mais **exception** `expected_min_pairs` non défini ⇒ photométrie instable ; two-pass CPU calcule des gains 0.949–0.997 (coverage min=0, max=38, mean≈7.6).
- GPU actuel (`zemosaic_workerGPU.log`) : seulement 65 paires, même exception `expected_min_pairs` ⇒ aucune correction intertile (`apply_photometric: no affine corrections available`) ; two-pass GPU reçoit une coverage saturée (min=0, max=66, mean≈56) et retourne des gains **tous à 1.0** → mosaïque corrompue.

---

## ✅ Plan de travail (avec cases à cocher)

### 1. Analyse différentielle

- [ ] Cloner / ouvrir deux vues du repo :  
  - une sur **`38c876a`** (référence)  
  - une sur la **branche actuelle**.
- [ ] Faire des diffs focalisés sur :
  - [ ] `zemosaic_worker.py` :
    - Phases 3, 4, 5, 6 ;
    - fonctions du type :
      - `_apply_phase5_post_stack_pipeline` :contentReference[oaicite:4]{index=4}  
      - `_apply_two_pass_coverage_renorm_if_requested`
      - `run_second_pass_coverage_renorm`
      - `assemble_final_mosaic_reproject_coadd`
      - `poststack_equalize_rgb` / traitement RGB-EQ dans les Master Tiles (cf. log `[RGB-EQ] poststack_equalize_rgb enabled=True, applied=True, gains=...`). :contentReference[oaicite:5]{index=5}  
  - [ ] `zemosaic_align_stack.py` et `zemosaic_align_stack_gpu.py` :
    - Toute différence sur les poids, l’ordre des opérations, les conversions RGB/BGR, les normalisations.
- [ ] Lister **précisément** toutes les différences **mathématiques / d’ordre d’opérations** entre `38c876a` et la branche actuelle pour la voie MASTER TILE → MOSAIC (CPU & GPU).
- [ ] Mettre en avant les écarts observés dans les runs récents vs golden :
  - [ ] Phase 5 intertile : en GPU, le log montre seulement 65 paires (vs 87 en CPU/golden) et aucune trace `apply_photometric`/gains par tuile (absents aussi en CPU actuel) alors qu’ils existent dans le log `38c876a` avec gain=0.86464 offset=-514.99.
  - [ ] Two-pass coverage : en GPU, la map de couverture de la passe 2 reste plate (cov min=max=1 pour canal 1) et les gains calculés sont tous 1.0, alors que la CPU calcule des gains 0.949–0.997 et le golden effectue la passe GPU sans ces plateaux.
  - [ ] Vérifier pourquoi la passe 2 GPU n’utilise qu’un seul canal en GPU (canal 1 GPU, canaux 2/3 CPU) et pourquoi les cov stats des canaux 2/3 s’étalent 0–15 alors que le canal 1 reste à 1.

### 2. Restauration STRICTE de la logique `38c876a` pour la voie Master Tile

Objectif : **copier / re-intégrer** la logique de `38c876a` dans la base actuelle, **sans “optimiser” ni “améliorer”**.

- [ ] Pour chaque fonction critique (exemples) :
  - `assemble_final_mosaic_reproject_coadd`
  - `_apply_phase5_post_stack_pipeline`
  - `_apply_two_pass_coverage_renorm_if_requested`
  - `run_second_pass_coverage_renorm`
  - tout bloc RGB-EQ / poststack_equalize_rgb
  - les fonctions de weighting / photometric solve en phase 5  
  → **Ramener la version `38c876a` telle quelle**, en l’adaptant juste si de **nouveaux paramètres** sont nécessaires pour la compatibilité avec la base actuelle.
- [ ] S’assurer que **la voie CPU et la voie GPU** appellent **les mêmes fonctions** de photométrie et de normalisation, dans le **même ordre**, avec les **mêmes paramètres** :
  - pas de code spécifique GPU qui change l’ordre des opérations ;
  - le GPU ne doit être qu’une **implémentation accélérée**, pas une logique différente.
- [ ] Vérifier en particulier :
  - [ ] qu’il n’y a **aucun mélange BGR/RGB** dans la voie GPU (OpenCV / CuPy) ;
  - [ ] que les gains RGB vérifiés dans le log de 38c876a (`[RGB-EQ] poststack_equalize_rgb ... gains=(1.000000,0.79...,1.03...)`) suivent la **même logique**. :contentReference[oaicite:6]{index=6}  
  - [ ] que l’intertile photometric solve est exécuté (paires ~87, `apply_photometric` avec gains/offsets) en CPU et GPU comme dans le log `38c876a`.
  - [ ] que la passe two-pass GPU produit une couverture non triviale (pas cov=1) et des gains non figés à 1.0, avec la même stratégie multi-canaux que la CPU/golden.
- [ ] S’il y a des nouveaux blocs “géniaux” ajoutés après `38c876a` qui modifient l’algorithme (streaming, nouvelles ponderations, etc.) :
  - [ ] **Les désactiver** pour la voie Master Tile “classique”.
  - [ ] Ou les garder **uniquement** si leur présence ne change **strictement rien** au résultat quand ils sont désactivés (i.e. même résultat qu’en 38c876a avec les mêmes settings).

### 3. Respect de la séparation SDS / classique

- [ ] Ne pas modifier la logique SDS (mosaïque d’abord, stacking ensuite, mega-tiles, Phase 4.5, etc.).
- [ ] S’assurer que les éventuels drapeaux / chemins “SDS vs non-SDS” ne se mélangent pas avec la voie Master Tile classique.
- [ ] Vérifier que la restauration de la logique `38c876a` ne casse pas :
  - les callbacks SDS,
  - la télémétrie / ParallelPlan,
  - le GUI (Tk / Qt).

### 4. Validation par “golden run” (CPU & GPU)

On dispose d’un **golden run** enregistré dans `zemosaic_worker38c876a.log` (commit bon) qui donne une mosaïque finale `(3237, 2399, 3)` avec second pass GPU OK. :contentReference[oaicite:7]{index=7}  

- [ ] Ajouter (si possible) un petit script/tests qui :
  - [ ] recharge les **mêmes brutes** que celles du run de référence (voire dossier /example dans le projet);
  - [ ] lance un run **CPU only** (GPU désactivé) ;
  - [ ] lance un run **GPU activé**.
- [ ] Vérifier :
  - [ ] que les shapes finales sont identiques à celles du log de référence.
  - [ ] que les logs clés sont présents et cohérents :
    - `run_info_phase5_started`, `assemble_info_finished_reproject_coadd`, `TwoPass` GPU, `run_info_phase5_finished_reproject_coadd`, etc. :contentReference[oaicite:8]{index=8}  
  - [ ] que les gains / target du `[RGB-EQ] poststack_equalize_rgb` sont du même ordre de grandeur que dans le log de référence (les valeurs exactes peuvent diverger un peu, mais **pas** au point de donner une dérive verte).
- [ ] Ajouter, si possible, un test automatique de comparaison CPU vs GPU :
  - [ ] calculer la **différence absolue moyenne** entre sortie CPU et sortie GPU ;
  - [ ] vérifier qu’elle est << dynamique de l’image (par ex. RMS < 1% de la plage) ;
  - [ ] échouer le test si la différence est trop grande.

### 5. Robustesse et non-régression

- [ ] Vérifier que le code compile et tourne sans erreur en :
  - [ ] mode Tk (GUI classique),
  - [ ] mode Qt,
  - [ ] mode CLI si disponible.
- [ ] S’assurer que :
  - [ ] aucun paramètre GUI n’a été cassé (GPU toggle, etc.).
  - [ ] les logs sont toujours produits dans `zemosaic_worker.log` (et éventuellement dans le dossier config utilisateur). :contentReference[oaicite:9]{index=9}  
- [ ] Documenter dans les commentaires principaux :
  - [ ] Quel bloc a été **restauré** à l’identique de `38c876a`.
  - [ ] Qu’il ne faut pas modifier ces blocs sans tests de non-régression CPU/GPU.

---

## ⚠️ Contraintes importantes

- **Pas de refactor “cosmétique”** sur ces blocs : priorité à la **stabilité** et à la **reproductibilité**.
- Ne **pas** changer les signatures publiques exposées au GUI, sauf nécessité absolue.
- Si une adaptation est vraiment nécessaire (par ex. ajout d’un paramètre demandé par d’autres parties du code), l’implémenter de façon à ce que, pour les paramètres utilisés dans ce projet, le comportement soit **identique** à `38c876a`.

---

## ✅ Critères de succès

- [ ] Avec le même jeu de données, la voie CPU produit une image visuellement équivalente à l’ancienne “image 3” (pas de bandes de jointure visibles, pas de dérive de couleur).
- [ ] Avec les mêmes paramètres, la voie GPU produit une image **indiscernable** de la voie CPU (à flot près).
- [ ] Les logs montrent la même séquence de phases / callbacks que dans `zemosaic_worker38c876a.log`.
- [ ] Aucun crash ni régression SDS ni GUI.
