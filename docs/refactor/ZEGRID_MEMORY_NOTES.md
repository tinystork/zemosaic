# ZeGrid — budget RAM & stacking canonique à mémoire bornée (note de design, DRAFT)

Statut : **note de réflexion**, pas une mission. À discuter au retour de ZM-ZEGRID-R2.
Origine : contrainte réelle tinydebian (7,5 GiB RAM, ~1–1,5 GiB dispo). Décision Tristan :
« problème à résoudre de toute façon, soit en réduisant la taille des mini-tiles adaptée à la
RAM, soit autre mécanisme ». Cette note liste les pistes, sans rien implémenter.

## 1. Où est réellement le pic mémoire

- **Lecture source : déjà bornée.** Le memmap / `hdu.section` (R1 `execution.py`) lit chaque
  brute par tranches ; aucune frame entière n'est chargée inutilement. Ce n'est **pas** le goulot.
- **Le pic est dans le moteur canonique.** `prepare_canonical_inputs` matérialise
  `out_images = np.empty((N, H, W, C), float32)`, plus les taper float64, plus les intermédiaires
  de rejet, plus les copies normalisées/per-frame. C'est **O(N × H × W)**, indépendant du memmap
  d'entrée. Mesuré : ~855 MiB de RSS pour N=7 sur un patch 496×835 (dont ~750 MiB de socle
  Python/Astropy/SciPy).
- **L'ancien chunking d'expositions est INVALIDE scientifiquement** : médiane par chunks ≠ médiane
  globale (témoin R0). Ne pas le réactiver pour économiser la RAM.

## 2. Levier 1 (immédiat, sans toucher SCI-05) — layout adapté à la RAM

- Propriété clé : le **travail total** est ~invariant à la taille des cellules (à l'overhead de
  halo près), alors que le **pic par cellule** ∝ `N_cell × patch_area`.
- Donc : réduire les cellules réduit le pic (~quadratique) **sans** réduire la science totale
  produite — c'est un levier *mémoire*, pas un levier *calcul*.
- Politique : choisir Nx/Ny (ou une dimension cible) pour que
  `pic_estimé = N_cell_estimé × patch_area × coeff_octets` ≤ `budget_RAM − marge_sécurité`,
  avec `coeff_octets` calibré par mesure (≈ 13–20 o / frame / pixel selon les plans matérialisés).
  `N_cell` dépend lui-même de la taille de cellule (overlap) → itérer 1–2 fois.
- **Planchers scientifiques obligatoires** (ne pas descendre plus bas) : aire de patch minimale
  (statistiques de rejet/normalisation), N minimal exploitable, overhead de halo maximal.
  Si le budget ne permet pas de respecter les planchers → le dire explicitement, jamais dégrader
  en silence.
- Coûts/handicaps : plus de cellules → plus d'overhead de halo (R0 : jusqu'à ~35 % en 9×7/h32) et
  plus de coûts fixes par cellule ; N plus faible → rejet moins robuste (low-N) ; référence/
  exclusions propres à chaque cellule (déjà le cas).
- Clé de config naturelle : un budget `grid_*_ram_mb` (le concept existait historiquement),
  lu au lancement, + marge fixe.

## 3. Levier 2 (durable, exact) — exécution canonique streaming en deux passes

Idée : rendre le moteur canonique **à mémoire bornée sans changer la science**, en exploitant la
structure des opérations.

- Les opérations canoniques se scindent en :
  1. **statistiques globales par frame** : coefficients de normalisation (régression linéaire sur
     le masque commun ref↔frame), σ/bruit, FWHM, choix de référence, exclusions ;
  2. **opérations par pixel à travers les frames** : rejet (kappa/winsor), combine, accumulation
     de support W1/W2/N_eff, taper.
- **Passe 1 (streaming sur tuiles spatiales)** : pour chaque frame, accumuler des statistiques
  suffisantes (sommes pour la régression vs réf, σ, comptes de validité) sans garder la frame
  entière → coefficients, poids, comptes, référence, exclusions.
- **Passe 2** : streamer des tuiles spatiales ; ne garder que `N × tuile_area` en mémoire ;
  appliquer les coefficients **déjà calculés** (pointwise), faire rejet/combine/support par pixel,
  écrire la tuile de sortie.
- **Taper** : EDT par frame avec un halo de `feather_px` → exact sur la tuile (l'EDT n'a qu'une
  portée locale bornée).
- Bilan mémoire : **O(N × tuile_area)** au lieu de **O(N × patch_area)** ; résultats numériques
  **identiques** (mêmes formules, mêmes sommes) — c'est une extension d'API, pas un changement de
  contrat scientifique. Frontière CPU **et** GPU (le pic GPU suit la même logique).
- Coût : une passe supplémentaire sur les données alignées (ou calcul de la passe 1 pendant
  l'alignement) + une **API SCI-05 streaming** dédiée, à valider bit-équivalent contre le moteur
  actuel sur la suite existante.

## 4. Recommandation

- **Court terme** : Levier 1 comme politique explicite « RAM-aware » (budget + marge + planchers
  scientifiques), mesurée ; c'est peu de code et ça débloque les gros champs sur tinydebian.
- **Durable** : Levier 2 comme **mission séparée d'extension SCI-05** (API canonique exacte à
  mémoire bornée), que ZeGrid consommera ensuite. À faire passer par gate humain (c'est de la
  science canonique).
- **En attendant** : Windows reste un fallback valide pour une cellule géante unique.

## 5. Questions ouvertes

- Sélection de référence en streaming : simple (il suffit des comptes de validité en passe 1).
- Le rejet low-N reste inchangé (par pixel à travers N) — vérifier la parité.
- Calibration du coefficient mémoire par backend (CPU/GPU) et par dtype.
- La passe 1 doit-elle réutiliser les frames alignées mises en cache (disque) plutôt que
  reprojeter deux fois ?
- Seuil exact des planchers scientifiques (aire min, N min, overhead halo max) : à mesurer, pas à
  graver.

---

## 6. `linear_fit` à l'échelle MiniTile — avantage qualitatif ? (mesuré 2026-10-06)

Question Tristan : « considérant la taille des minitiles, linear_fit présente-t-il un avantage qualitatif ? »

Mesure sur un vrai contributeur quasi-plein de `r0000c0001` (172 900 px communs, M106 20 s) :

| estimateur | pente a |
|---|---|
| OLS brut sur le masque commun (ce que le moteur calcule) | **0,6137** |
| Pearson r (même masque) | 0,553 |
| OLS sur les 10 % / 5 % / 1 % les plus brillants | 0,913 / 0,921 / 0,926 |
| lissé σ=4 | 0,9256 |

Lecture :

1. **Le vrai terme multiplicatif est petit** (~0,93, soit ~7 %) entre deux poses de même exposition.
2. **L'estimateur par patch est fortement dilué par le bruit** : `E[â] ≈ a·r`, donc à r≈0,55 la pente brute vaut ~0,61 au lieu de ~0,93. L'erreur injectée (×1,63 sur le flux de la frame) est **~9× plus grande que l'effet corrigé (~7 %)**.
3. Le garde-fou `[0,25–4,0]` ne protège pas : une pente brute de 0,61 serait **acceptée** et appliquée.

Conclusion : **à l'échelle MiniTile, `linear_fit` n'apporte pas d'avantage qualitatif fiable ; en bas SNR il dégrade la photométrie là où `sky_mean` (a=1, à ~7 % de la vérité ici) est sain.**

### Quand le terme multiplicatif devient réellement utile

- quand les **expositions/gains diffèrent** entre frames (a ≈ rapport d'exposition, ex. ×2) : là a=1 est franchement faux ;
- quand le **domaine d'estimation est profond / haut SNR** (beaucoup de frames par pixel, ou grand champ commun) : l'estimation converge ;
- quand la **transparence varie** sensiblement sur la session.

### Le vrai remède n'est pas le choix par patch, mais le DOMAINE

**La normalisation est une propriété de la FRAME (ou du couple frame↔référence), pas du patch.** Estimer un a par MiniTile est (a) mal conditionné sur les recouvrements minces et (b) fait diverger les cellules voisines → **marches de gain inter-cellules** (le résidu de seam R2 en est partiellement la conséquence). Estimer **une fois par frame sur le plus grand domaine commun**, puis appliquer partout :

- meilleur conditionnement (masque plus grand, sous-ensemble haut SNR) ;
- **mêmes facteurs pour toutes les cellules** → les marches de gain de normalisation disparaissent du seam ;
- reste physiquement frame-global, pas patch-arbitraire.

Recommandation : garder `sky_mean` par défaut à l'échelle MiniTile ; traiter le terme multiplicatif comme un **sujet frame-level** à décider (et seulement nécessaire si les expositions diffèrent). La future mission SCI-05 devrait donc être reformulée : **« quel domaine de normalisation » (frame vs patch)**, plutôt que « réparer le MAD ». Si le domaine devient frame-level, l'affine redevient estimable et la question « garder ou supprimer linear_fit » se repose proprement.
