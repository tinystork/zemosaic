
## `followup.md`

### Situation

Cette fiche est à utiliser **après** un premier passage de Codex pour :

* Vérifier que la détection de backend est bien en place.
* Corriger les éventuels problèmes (import, typage, visibilité du bouton).
* Affiner les messages et le logging si nécessaire.

---

### Check-list rapide

1. **[x] Imports**

   * `zemosaic_gui_qt.py` contient bien :

     ```python
     from zemosaic_utils import get_app_base_dir  # type: ignore
     from zemosaic_time_utils import ETACalculator, format_eta_hms
     from path_helpers import safe_path_isdir
     ```

   * Et le fallback dans le `except` définit **aussi** un `safe_path_isdir()` minimaliste.

   * Aucune nouvelle dépendance externe n’a été ajoutée.

2. **[x] Helper `_detect_analysis_backend`**

   * La fonction existe, retourne un tuple `(backend, root)` avec :

     * `backend` ∈ `{"none", "zeanalyser", "beforehand"}`.
     * `root` est un `Path` ou `None`.
   * Elle utilise `get_app_base_dir().parent` comme racine pour chercher `zeanalyser` et `seestar/beforehand`.
   * La priorité `zeanalyser` > `beforehand` est bien respectée.

3. **[x] Intégration dans `ZeMosaicQtMainWindow`**

   * L’init de la classe contient :

     ```python
     self.analysis_backend: AnalysisBackend = "none"
     self.analysis_backend_root: Optional[Path] = None
     ```

   * Après chargement de la config, on a :

     ```python
     self.analysis_backend, self.analysis_backend_root = _detect_analysis_backend()
     ```

   * Un attribut `self.analysis_button: QPushButton | None` est défini (soit dans `__init__`, soit implicitement mais cohérent).

4. **Bouton « Analyse »**

   * Dans `_build_command_row` :

     * Le bouton est créé **uniquement** si `self.analysis_backend != "none"`.
     * L’ordre des autres boutons est conservé : `Filter`, `Start`, `Stop` (puis éventuellement `Analyse`).
   * Le bouton est connecté à `self._on_analysis_clicked`.

5. **Handler `_on_analysis_clicked`**

   * La méthode existe, ne modifie pas le worker, ne lance pas de pipeline.
   * Elle :

     * Vérifie l’état du backend.
     * Logge un message clair dans la zone de log (par ex. `"[INFO] [Analyse] Backend=zeanalyser, root=..."`).
     * Affiche une `QMessageBox.information` différente pour ZeAnalyser / Beforehand.
   * En cas de backend `none` ou `root is None`, la méthode :

     * Désactive le bouton et/ou affiche un message type “no backend available”.
     * Ne lance aucune exception.

6. **[x] Internationalisation**

   * Dans `zemosaic_localization.py` :

     * La clé `qt_button_analyse` est présente dans EN/FR (et autres langues si applicable).
   * Le texte par défaut dans `_tr("qt_button_analyse", "Analyse")` est correct.

7. **[x] Tk GUI**

   * Vérifier que **`zemosaic_gui.py` n’a pas été modifié**.
   * Aucun nouveau bouton / logique d’analyse n’y figure.

---

### Diagnostics si quelque chose ne va pas

* **Le bouton n’apparaît jamais :**

  * Vérifier que la détection utilise bien `get_app_base_dir().parent`.
  * Logger temporairement dans `_detect_analysis_backend()` :

    * `base_dir`, `toolbox_root`, `zeanalyser_dir`, `beforehand_dir`, et résultat.
  * Vérifier l’orthographe exacte des dossiers (`zeanalyser`, `seestar`, `beforehand`).

* **Crash au démarrage (ImportError lié à `path_helpers`) :**

  * S’assurer que `safe_path_isdir` est bien défini dans le bloc `except` de l’import.
  * S’assurer qu’aucun autre code ne dépend d’une version “complète” de `path_helpers`.

* **Crash à l’appel de `_on_analysis_clicked` :**

  * Vérifier les types (`Optional[Path]`) et les checks `if backend == "none" or root is None`.
  * S’assurer que `self._append_log` est appelé avec une simple chaîne (sans formatage exotique).

---

### Évolutions possibles (pour une mission ultérieure)

* Mettre à jour `_on_analysis_clicked` pour lancer automatiquement :

  * ZeAnalyser via `subprocess.Popen([...])` ou une API Python si elle existe.
  * Beforehand via un script ou module dédié.

* Passer en paramètre à l’outil d’analyse :

  * Le chemin de la mosaïque finale.
  * Le WCS global.
  * Le dossier d’output / input courant.

* Ajouter des options avancées dans la GUI (par ex. choix du backend si les deux sont présents, toggle “auto-open analyser at end of run”, etc.).
