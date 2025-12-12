# 🎯 Mission — Diagnostic précis du décalage vert (mode Classic)

## Objectif
Identifier **l’étape exacte** du pipeline Classic où le canal vert (G)
commence à diverger statistiquement par rapport à R et B.

Aucun refactor.
Aucune modification de logique.
**Logs DEBUG uniquement**, ciblés et comparables entre Classic / SDS.

---

## Contraintes strictes
- ❌ Ne modifier aucun calcul existant
- ❌ Ne pas changer l’ordre des phases
- ❌ Pas de normalisation supplémentaire
- ✅ Ajouter uniquement des logs conditionnés au niveau DEBUG
- ✅ Logs compacts, lisibles, comparables

---

## Pré-requis
Le niveau de logging sélectionné dans le GUI Qt (`Logging level`)
doit être **propagé correctement jusqu’au logger du worker**.

---

## Outil de log à utiliser
Utiliser exclusivement la fonction existante :

_dbg_rgb_stats(
label: str,
rgb: np.ndarray,
coverage: np.ndarray | None = None,
alpha: np.ndarray | None = None,
logger: logging.Logger
)

markdown
Copier le code

Cette fonction calcule :
- min / mean / median par canal
- ratio G/R et G/B
- stats pondérées par coverage si fourni
- uniquement sur pixels valides

---

## 🔍 Phase 3 / 3.x — Stack des master tiles (baseline)

### Objectif
Prouver noir sur blanc que la couleur est saine **avant toute mosaïque**.

### Points de log (DEBUG uniquement)
Pour un petit échantillon de tiles (déjà sélectionné par `_select_debug_tile_ids`) :

- [x] Avant `stack_core`
- [x] Après `stack_core`
- [x] Après `_poststack_rgb_equalization` (si appelée)

### Labels à utiliser
- `P3_pre_stack_core`
- `P3_post_stack_core`
- `P3_post_poststack_rgb_eq`

---

## 🔥 Phase 4 / 4.x — Assemblage mosaïque (ZONE CRITIQUE N°1)

### Objectif
Détecter l’apparition du déséquilibre lors du passage tile → plan global.

### Points de log
1. **Avant fusion**
   - [x] `P4_pre_merge_rgb`

2. **Après fusion brute**
   - [x] `P4_post_merge_rgb`

3. **Après application coverage / NaN**
   - [x] `P4_post_merge_valid_rgb`
   - [x] fournir `coverage=final_mosaic_coverage`

4. **Moyenne pondérée par coverage**
   - [x] via `_dbg_rgb_stats` (si coverage présent)

---

## 🔥🔥 Phase 5 — Post-processing global (ZONE CRITIQUE N°2)

### Objectif
Identifier une normalisation RGB globale incorrecte.

### Points de log
1. Avant toute égalisation globale
   - [x] `P5_pre_rgb_equalization`

2. Après `_apply_final_mosaic_rgb_equalization`
   - [x] `P5_post_rgb_equalization`

### Si une égalisation RGB est appliquée
Logger explicitement :
- cibles par canal
- facteurs multiplicatifs appliqués
- masque utilisé (si existant)

⚠️ Si `ratio_G_R` ou `ratio_G_B` dérive ici → **coupable identifié**

---

## Phase 6–7 — Export / clamp (secondaire)

### Objectif
Exclure définitivement une cause d’export.

### Logs
- dtype avant export
- min / max par canal avant clamp
- conversion float → uint

Labels :
- `P6_pre_export`
- `P6_post_export`

---

## Critère de succès
Identifier **la première phase** où :
ratio_G_R ≠ ~1
ratio_G_B ≠ ~1