# ZeMosaic — Patch ciblé (Qt) : Borrowing v1 réellement actif + log fichier restauré

## Mission

Appliquer un patch **minimal et chirurgical** pour :

1. [x] rendre **Borrowing v1** réellement opérant en Qt (en lisant correctement les centres `center_ra_deg/center_dec_deg`),
2. [x] restaurer l’écriture du fichier **`zemosaic_filter.log`** par `zemosaic_filter_gui_qt.py` (FileHandler manquant),
3. [x] ajouter **2 compteurs de preuve** (debug) permettant de valider en un run que Borrowing v1 “voit” bien les données.

## Contexte

* Le hook d’appel à `apply_borrowing_v1(...)` est présent dans `zemosaic_filter_gui_qt.py`.
* Le log du worker ne montre pas d’effet mesurable ; cause probable : `apply_borrowing_v1` n’extrait pas les centres des images lorsque les champs s’appellent `center_ra_deg` / `center_dec_deg`.
* `zemosaic_filter_gui_qt.py` supprime `zemosaic_filter.log` au démarrage mais **ne configure pas de FileHandler**, donc aucun fichier n’est recréé.

## Contraintes (garde-fous)

* [x] **Ne pas** toucher au worker, à `stack_core`, à la reprojection, ni au pipeline mosaïque.
* [x] **Ne pas** ajouter de nouveau système de logs : seulement un `FileHandler` standard Python logging.
* [x] **Ne pas** ajouter d’UI. Le dropdown de niveau de log existe déjà : on ne le recrée pas.
* [x] Patch limité aux fichiers :

  * [x] `zemosaic_utils.py`
  * [x] `zemosaic_filter_gui_qt.py`

## Objectifs de preuve (doivent apparaître dans les logs)

* [x] Une ligne `INFO` confirmant que le FileHandler est en place :

  * [x] `Filter log file handler enabled: <path>`
* [x] Une ligne `INFO` confirmant que Borrowing a été exécuté et avec quelles stats :

  * [x] `Borrowing v1 applied: groups=... borrowed_unique=... borrowed_total=...`
* [x] Deux compteurs `DEBUG` (ou `INFO_DETAIL` si tu as cette granularité) dans Borrowing :

  * [x] nombre d’images ayant un centre valide
  * [x] nombre de groupes ayant un centre valide

## Tâches

### 1) `zemosaic_utils.py` — rendre `_extract_center_tuple` compatible Qt

Trouver la fonction `_extract_center_tuple(img)` (utilisée par `apply_borrowing_v1`).

Ajouter la prise en charge des clés suivantes **sans casser l’existant** :

* [x] `center_ra_deg` / `center_dec_deg`
* [x] `center_ra` / `center_dec` (si présentes)

Règles :

* [x] Retourner un tuple `(float(x), float(y))` si possible.
* [x] Si valeurs manquantes/NaN/non castables : retourner `None`.
* [x] Conserver le comportement actuel pour les clés existantes (`RA/DEC`, `ra/dec`, etc.).

### 2) `zemosaic_utils.py` — 2 compteurs de preuve dans `apply_borrowing_v1`

Dans `apply_borrowing_v1(...)`, ajouter des compteurs simples :

* [x] `valid_image_centers` = nombre d’images ayant un centre extractible
* [x] `valid_group_centers` = nombre de groupes dont le centre a pu être calculé

Logguer une seule fois au début (après extraction) :

* [x] `logger.debug("Borrowing v1: valid_image_centers=%d valid_group_centers=%d", ...)`

IMPORTANT :

* [x] Pas de spam de log.
* [x] Aucune modification des règles (border/quota/guard) dans ce patch.

### 3) `zemosaic_filter_gui_qt.py` — restaurer `zemosaic_filter.log`

Ajouter une fonction minimaliste qui assure qu’un `FileHandler` existe.

Exigences :

* [x] Réutiliser `logging` standard.
* [x] Créer le fichier log au même endroit que le script (même stratégie que `_reset_filter_log`).
* [x] Éviter les doublons de handlers (ne pas ajouter 2 fois).

Pseudo-structure (adapter au style du code) :

* [x] `def _ensure_filter_file_logger() -> None:`

  * [x] `log_path = Path(__file__).with_name("zemosaic_filter.log")`
  * [x] `root = logging.getLogger()` ou logger module
  * [x] vérifier si un handler FileHandler existe déjà sur ce chemin
  * [x] sinon : créer FileHandler + formatter + addHandler
  * [x] log `INFO` : `Filter log file handler enabled: ...`

Appel :

* [x] appeler `_ensure_filter_file_logger()` **au démarrage**, juste après `_reset_filter_log()` (ou au début du point d’entrée Qt), avant que l’UI lance les opérations.

### 4) `zemosaic_filter_gui_qt.py` — preuve d’appel Borrowing

Juste après l’appel à `apply_borrowing_v1(...)`, ajouter un log `INFO` compact :

* [x] `Borrowing v1 applied: groups=... borrowed_unique=... borrowed_total=...`

Si `apply_borrowing_v1` retourne des stats dict, les utiliser.
Sinon, utiliser des valeurs par défaut sûres.

## Tests (un seul run)

1. Lancer un run legacy coverage-first via Qt (le même dataset que tes captures).
2. Vérifier que `zemosaic_filter.log` est créé et contient des lignes.
3. Chercher dans `zemosaic_filter.log` :

   * `Filter log file handler enabled:`
   * `Borrowing v1:` (valid_image_centers / valid_group_centers)
   * `Borrowing v1 applied:`

## Critère de succès

* Le fichier `zemosaic_filter.log` existe et se remplit.
* Borrowing v1 prouve qu’il voit des centres valides (compteurs non nuls) et qu’il s’exécute (ligne applied).
* Aucun changement fonctionnel ailleurs.
