✅ agent.md — Coverage-First Hard Merge (Seestar-friendly)
🎯 Mission

Optimiser la phase de “Coverage-First Preplan” dans zemosaic_filter_gui.py (Qt) afin de réduire le nombre de micro-groupes (1–10 frames) générés lorsque le champ est très recouvrant (ex : Seestar S50).
⚠️ Sans modifier la logique SDS ou le reste du pipeline.

Objectif :
→ produire moins de groupes, mais plus robustes, avec un SNR interne plus homogène,
→ tout en préservant les groupes réellement isolés.

📌 Règles du “Hard Merge”

Le merge est strictement local et non destructif :

1. Groupes éligibles au merge

Un groupe est candidat s’il vérifie :

group.size < merge_threshold

valeur par défaut : 10

rendre la valeur configurable via solver_settings.py ou un paramètre interne au module

groupe non vide et non SDS préalloué

2. Critère spatial (obligatoire)

Un micro-groupe A ne peut être fusionné qu’avec un groupe B si :

Distance angulaire des centres < FoV × 1.2
(déjà disponible via footprint RA/Dec)
OU

Footprints RA/Dec qui se recoupent réellement
(rectangle intersection stricte)

⚠️ Si A n’a aucun voisin qui respecte cela → NE PAS fusionner.
→ C’est ce qui protège les paquets éloignés comme dans ta capture.

3. Critère de taille (cap & overcap)

Si spatialement admissible, fusion autorisée seulement si :

size(A) + size(B) ≤ max_raw_per_master_tile × (1 + overcap_allowance_fraction)

Remarques :

utiliser exactement la même valeur slider “overcap allowance (%)”

transformer 10% → 0.10 pour la formule

refuser toute fusion qui dépasse ce plafond

4. Merge unique

Un micro-groupe ne doit être fusionné qu’une seule fois, pour éviter les chaînes infinies :

A → fusionne dans le meilleur candidat B

A disparaît

B est mis à jour

A n’est jamais revu

5. Ordre de fusion

Fusionner dans cet ordre :

micro-groupes les plus petits en premier

puis ceux un peu plus gros
Cela maximise les fusions réussies.

6. Logging

Ajouter des lignes dans le logger :

[HARD-MERGE] Merged group #A (size=4) → group #B (size=12), dist=0.42°, new_size=16


Si rejet :

[HARD-MERGE] Skip group #A : no eligible neighbour
[HARD-MERGE] Skip merge #A→#B : would exceed cap (22 > 20)

7. Aucun autre impact

Ne rien modifier à :

SDS

Auto-tile heuristics

Zesupadupstack

la logique de coverage map

lecropper

le code Phase 5 et Phase 3

Organiser le code proprement dans une fonction dédiée :

_apply_hard_merge(groups, settings, logger)

à placer dans zemosaic_filter_gui.py, juste après _merge_small_groups() mais appelée après l’étape de preplan, avant affichage GUI et serialization dans overrides_state.preplan_master_groups.

📁 Fichiers à modifier

zemosaic_filter_gui.py (principal)

éventuellement :

solver_settings.py (clé config merge_threshold si besoin)

zemosaic_utils.py (helper rectangle intersection si utilitaire manquant)

🧪 Tests à passer
Cas 1 — Seestar ultra-recouvrant (ex : 3500 frames)

Entrée : ton dataset typique avec 180+ groupes.
Attendu :

180 → ~30–50 groupes (ordre de grandeur)

tous les groupes restants ≥ 15–20 frames

logs de hard-merge présents

aucun dépassement cap

Cas 2 — Champs éclatés (comme ta 2ᵉ capture)

Entrée : 4–6 clusters éloignés.
Attendu :

aucune fusion

logs : “no eligible neighbour”

nombre de groupes identique à avant le patch

Cas 3 — Cap faible / overcap faible

Attendu :

fusions refusées proprement

logs explicites

Cas 4 — Cap élevé / overcap élevé

Attendu :

fusions plus agressives mais toujours locales

aucune fusion entre zones distantes

🔒 Contraintes

Ne toucher AUCUNE logique SDS

Ne rien changer à la structure des master tiles

Aucun impact sur le pipeline standard

Backward compatible

Codex doit produire un patch propre, clair, bien commenté

Le comportement batch size = 0 / >1 ne doit jamais être altéré

