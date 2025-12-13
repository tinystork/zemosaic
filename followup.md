# Follow-up — Checklist patch ciblé (Qt)

## 0) Garde-fous

* [x] Ne modifier **que** `zemosaic_utils.py` et `zemosaic_filter_gui_qt.py`.
* [x] Ne toucher ni au worker, ni au pipeline mosaïque, ni à grid_mode.
* [x] Ne pas ajouter de nouvelle UI.
* [x] Ne pas changer les règles Borrowing (quota / border / guard) dans ce patch.

---

## 1) `zemosaic_utils.py` — extraction centres compatible Qt

* [x] Localiser `_extract_center_tuple(img)`.
* [x] Ajouter support des clés :

  * [x] `center_ra_deg` + `center_dec_deg`
  * [x] `center_ra` + `center_dec` (si présentes)
* [x] Conversion robuste : `float(...)` ; si échec → `None`.
* [x] Ne pas casser les chemins existants (`RA/DEC`, `ra/dec`, etc.).

### Mini-test mental

* [x] Si `img` contient `center_ra_deg=150.123` et `center_dec_deg=2.456`, la fonction renvoie `(150.123, 2.456)`.
* [x] Si `center_ra_deg` est absent ou non convertible, renvoie `None`.

---

## 2) `zemosaic_utils.py` — 2 compteurs de preuve

Dans `apply_borrowing_v1(...)` :

* [x] Compter `valid_image_centers`.
* [x] Compter `valid_group_centers`.
* [x] Log **une seule fois** :

  * [x] `logger.debug("Borrowing v1: valid_image_centers=%d valid_group_centers=%d", ...)`

---

## 3) `zemosaic_filter_gui_qt.py` — recréer `zemosaic_filter.log`

* [x] Confirmer qu’il existe `_reset_filter_log()` (supprime le fichier).
* [x] Ajouter `_ensure_filter_file_logger()`.
* [x] Dans `_ensure_filter_file_logger()` :

  * [x] `log_path = Path(__file__).with_name("zemosaic_filter.log")`
  * [x] Trouver le logger cible (root ou module) utilisé par le reste du fichier.
  * [x] Détecter si un FileHandler sur ce chemin est déjà présent.
  * [x] Sinon : créer FileHandler + formatter + `addHandler`.
  * [x] Log `INFO`: `Filter log file handler enabled: ...`
* [x] Appeler `_ensure_filter_file_logger()` au démarrage (juste après `_reset_filter_log()` ou dans le point d’entrée).

### Validation

* [ ] Lancer l’app, vérifier que `zemosaic_filter.log` est créé.
* [ ] Vérifier une ligne : `Filter log file handler enabled:`.

---

## 4) `zemosaic_filter_gui_qt.py` — preuve d’appel Borrowing

* [x] Localiser l’appel : `final_groups, _borrow_stats = apply_borrowing_v1(...)`.
* [x] Juste après, ajouter log `INFO` :

  * [x] `Borrowing v1 applied: groups=%d borrowed_unique=%d borrowed_total=%d`
* [x] S’assurer que `_borrow_stats` est utilisé si présent, sinon fallback à 0.

---

## 5) Test unique (run complet)

* [ ] Run legacy coverage-first via Qt (dataset M106).
* [ ] Dans `zemosaic_filter.log`, confirmer :

  * [ ] `Filter log file handler enabled:`
  * [ ] `Borrowing v1: valid_image_centers=... valid_group_centers=...`
  * [ ] `Borrowing v1 applied:`

---

## 6) Critère de succès

* [ ] `zemosaic_filter.log` existe et s’alimente.
* [ ] Borrowing v1 montre qu’il “voit” des centres Qt (compteurs > 0).
* [ ] Aucun changement fonctionnel ailleurs.
