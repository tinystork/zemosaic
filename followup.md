# followup.md

## Ce que tu dois livrer (Codex)
- Un patch minimal centré sur le **Master tile quality gate** (ZeQualityMT) qui rejette :
  1) master tiles trop petites / dégénérées,
  2) master tiles à “edge blow-up” (bords outliers extrêmes),
  3) (si déjà dans la logique) tuiles trop “vides” selon métriques existantes,
  sans changer l’UI.
- Important : `NBR` seul ne suffit pas (poids faible dans le score). Les cas `small_dim` / `edge_blowup` doivent être des **hard rejects** (acceptation forcée à FALSE + override bloqué, sans dépendre du score).

## Détails d’implémentation recommandés
### 1) Détection edge blow-up (robuste)
- Construire `plane` (intensité):
  - RGB → `np.nanmax(np.abs(data[..., :3]), axis=2)`
  - mono → `np.abs(data)`
- Réutiliser les masques déjà construits dans `quality_metrics` : `edge_mask` (ring) et `center_mask` (core) — `b` est déjà clampé.
- Garde-fou “slice vide / all-NaN” :
  - si `plane[center_mask]` ou `plane[edge_mask]` n’a **aucune** valeur finie → **hard reject** `reason=no_finite_core_or_edge`
- Calcul :
  - `core_p99 = np.nanpercentile(plane[center_mask], 99)`
  - `edge_p99 = np.nanpercentile(plane[edge_mask], 99)`
- Déclencheur conservateur (ne pas rejeter pour une simple étoile brillante au bord, viser les artefacts extrêmes) :
  - `edge_p99 > max(core_p99, eps) * 1e6`
- Coupe-circuit “cap absolu” (optionnel mais très utile) :
  - si `np.nanmax(plane) > 1e25` → **hard reject** `reason=abs_cap_exceeded`
- Si déclenché : **hard reject**
  - exposer des métriques (`core_p99`, `edge_p99`, `edge_ratio`) et un flag (`hard_reject=1`, `reason=edge_blowup` via des clés **stables et non-renommables**),
  - et dans la décision d’acceptation :
    - `zequalityMT.py:run_cli`
    - `zemosaic_worker.py:_evaluate_quality_gate_metrics`
    `hard_reject=1` ⇒ `accepted=False` **quel que soit** le threshold (score éventuellement élevé uniquement pour debug).

### 2) Étendre NBR (“bad pixels on edges”)
- Bad pixel = non-fini OR <= thr_zero OR >= high_cap
- `high_cap = max(core_p99, eps) * 1e8`
- NBR = fraction de bad dans ring/edges
- Utiliser `k_sigma` si déjà dans le code (sinon rester simple avec p99 ratio).

### 3) Rejet tuiles trop petites
- Dans le gate, si `min(h, w) < 128` → **hard REJECT**. Le calcul doit être `min(arr.shape[0], arr.shape[1])` pour ignorer le channel.
(seuil volontairement conservateur, mais **doit** au minimum attraper 32×32)

### 4) Override d’acceptation
- Si une fonction “accept_override” existe :
  - empêcher l’acceptation d’une tuile si `hard_reject=1` (en particulier `edge_blowup` / `small_dim`),
  - synchroniser la logique entre `zequalityMT.py` et `zemosaic_worker.py` (la règle est dupliquée côté worker).
- Rendre la règle explicite :
  - dans `zequalityMT.py:_accept_override` et `zemosaic_worker.py:_zequality_accept_override` : commencer par `if hard_reject: return False`

## Tests à exécuter
### Tests unitaires (si existants)
- Lancer la suite existante : `pytest -q` (ou commande projet).
- Ajouter (si possible) un test simple “synthetic tile” :
  - image noire + bande de bord à 1e12
  - vérifier que le gate la rejette.

### Test reproductible manuel
- Activer “Master tile quality gate” (le log doit montrer `quality_gate=True`) avec des paramètres typiques (ceux de l’UI existante) :
  - Edge band ~64
  - K-sigma ~2.5
  - Erode ~3–8 (max UI = 12)
- Rejouer le run sur un dataset contenant des tuiles toxiques.
- Vérifier :
  - fichiers déplacés dans sous-dossier “rejected”
  - plus de rectangles/cadres sur mosaïque finale
  - logs de rejet présents (inclure `shape=(H,W)` + `b=...` pour diagnostic immédiat)

## Garde-fous
- Ne pas modifier l’assemblage final (reproject/coadd).
- Ne pas ajouter de nouveaux paramètres UI.
- Ne pas modifier les comportements “batch size” existants.
- Toute modification doit être encapsulée dans le quality gate :
  - si gate OFF → aucune différence.
  - **Test manuel simple** : pour un même dataset, un run avec le code patché mais `quality_gate_enabled=False` doit produire exactement la même liste et le même nombre de master tiles qu'un run avec le code *avant* le patch. La vérification de la liste des tuiles suffit, pas besoin de comparer le FITS final.

## Definition of Done
- Les tuiles pathologiques (dégénérées ou bords explosés) sont rejetées par le gate.
- Le rendu final ne montre plus les cadres.
- Aucun impact visible sur les runs où ces cas n’apparaissent pas.
- Log clair et actionnable.

