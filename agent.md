
### Mission

Ajouter un bouton **« Analyse »** dans l’interface Qt de ZeMosaic (`zemosaic_gui_qt.py`) **sans toucher au worker / à la pipeline** :

* Le bouton **n’apparaît que si** un backend d’analyse est détecté dans l’arborescence :

  * **ZeAnalyser** : dossier `zeanalyser` présent dans le **répertoire parent** de `zemosaic`.
  * **Beforehand** : dossier `seestar/beforehand` présent sous le même parent.
* Si **les deux** existent, **ZeAnalyser est prioritaire**.
* Le clic sur le bouton n’a, pour l’instant, qu’un comportement minimal : log + message utilisateur indiquant quel backend a été détecté (l’intégration “profonde” viendra plus tard).
* La fonctionnalité est **limitée à la GUI Qt**. Le vieux GUI Tk **ne doit pas être modifié**.

---

### Contexte

* Le projet est organisé typiquement comme ceci :

  ```text
  .../zeseestarstacker/
      zemosaic/
          zemosaic_gui_qt.py
          zemosaic_utils.py
          ...
      zeanalyser/
          ...
      seestar/
          beforehand/
              ...
  ```

* `zemosaic_utils.get_app_base_dir()` renvoie le répertoire de base de ZeMosaic (le dossier `zemosaic`, même en mode PyInstaller).

* Le **parent** de ce répertoire est la racine “toolbox” ZeSeestarStacker dans laquelle se trouvent éventuellement `zeanalyser` et `seestar/beforehand`.

* Il existe déjà des helpers robustes pour les chemins dans `path_helpers.py` (`safe_path_isdir`).

Objectif : utiliser ces briques pour détecter automatiquement la présence d’un outil d’analyse et conditionner l’affichage du bouton dans `zemosaic_gui_qt.py`.

---

### Fichiers à modifier

1. `zemosaic_gui_qt.py`
2. (Optionnel, mais recommandé) `zemosaic_localization.py`
   → pour ajouter la clé de traduction du bouton « Analyse » si nécessaire.

> **Important :** ne pas toucher aux autres modules (worker, grid_mode, align, stack, etc.).
> Aucune modification de la logique scientifique / de la pipeline.

---

### Détails de l’implémentation

#### 1. Importer `safe_path_isdir`

Dans `zemosaic_gui_qt.py`, il y a déjà un bloc `try/except` en haut du fichier qui importe `get_app_base_dir` :

```python
try:
    from zemosaic_utils import get_app_base_dir  # type: ignore
    from zemosaic_time_utils import ETACalculator, format_eta_hms
except Exception:  # pragma: no cover - fallback when utils missing
    def get_app_base_dir() -> Path:  # type: ignore
        return Path(__file__).resolve().parent
```

À adapter comme suit :

* Ajouter l’import de `safe_path_isdir` dans le `try` :

```python
try:
    from zemosaic_utils import get_app_base_dir  # type: ignore
    from zemosaic_time_utils import ETACalculator, format_eta_hms
    from path_helpers import safe_path_isdir
except Exception:  # pragma: no cover - fallback when utils missing
    def get_app_base_dir() -> Path:  # type: ignore
        return Path(__file__).resolve().parent

    def safe_path_isdir(pathish: str | os.PathLike | None, *, expanduser: bool = True) -> bool:
        """Fallback minimaliste pour éviter un crash si path_helpers n'est pas dispo."""
        if pathish is None:
            return False
        try:
            text = os.path.expanduser(str(pathish)) if expanduser else str(pathish)
            return os.path.isdir(text)
        except Exception:
            return False
```

* `os` est déjà importé dans ce fichier ; sinon, l’ajouter en haut.

#### 2. Créer un petit helper de détection de backend

Toujours dans `zemosaic_gui_qt.py`, au niveau des helpers privés (par exemple à proximité de `_expand_to_path`), ajouter :

```python
from typing import Literal, Tuple

AnalysisBackend = Literal["none", "zeanalyser", "beforehand"]


def _detect_analysis_backend() -> Tuple[AnalysisBackend, Optional[Path]]:
    """
    Inspecte l'arborescence autour de ZeMosaic pour trouver un backend d'analyse.

    Logique :
      - base_dir = get_app_base_dir()  -> dossier zemosaic
      - toolbox_root = base_dir.parent
      - Si toolbox_root / "zeanalyser" est un dossier  -> "zeanalyser"
      - Sinon si toolbox_root / "seestar" / "beforehand" est un dossier -> "beforehand"
      - Sinon -> "none"
    """

    try:
        base_dir = get_app_base_dir()
    except Exception:
        return "none", None

    toolbox_root = base_dir.parent

    zeanalyser_dir = toolbox_root / "zeanalyser"
    beforehand_dir = toolbox_root / "seestar" / "beforehand"

    # Priorité à ZeAnalyser si les deux existent
    if safe_path_isdir(zeanalyser_dir):
        return "zeanalyser", zeanalyser_dir

    if safe_path_isdir(beforehand_dir):
        return "beforehand", beforehand_dir

    return "none", None
```

#### 3. Ajouter un état interne au `ZeMosaicQtMainWindow`

Dans `ZeMosaicQtMainWindow.__init__` :

* Initialiser deux nouveaux attributs d’instance :

```python
self.analysis_backend: AnalysisBackend = "none"
self.analysis_backend_root: Optional[Path] = None
```

* Après le chargement de la configuration et avant la construction des widgets (idéalement avant `_initialize_tab_pages()` ou équivalent), appeler le helper :

```python
self.analysis_backend, self.analysis_backend_root = _detect_analysis_backend()
```

L’idée : la détection est faite **une seule fois au démarrage** de la fenêtre.

#### 4. Créer le bouton « Analyse » dans la barre de commandes

Dans la méthode `_build_command_row` de `ZeMosaicQtMainWindow`, on a actuellement quelque chose du genre :

```python
def _build_command_row(self) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addStretch(1)
    self.filter_button = QPushButton(self._tr("qt_button_filter", "Filter…"))
    ...
    self.start_button = QPushButton(self._tr("qt_button_start", "Start"))
    self.stop_button = QPushButton(self._tr("qt_button_stop", "Stop"))
    ...
    row.addWidget(self.filter_button)
    row.addWidget(self.start_button)
    row.addWidget(self.stop_button)
    return row
```

À adapter comme suit :

1. Déclarer un `self.analysis_button: QPushButton | None` dans `__init__` (pour typage et clarté) :

   ```python
   self.analysis_button: QPushButton | None = None
   ```

2. Dans `_build_command_row` :

   ```python
   def _build_command_row(self) -> QHBoxLayout:
       row = QHBoxLayout()
       row.setContentsMargins(0, 0, 0, 0)
       row.addStretch(1)

       self.filter_button = QPushButton(self._tr("qt_button_filter", "Filter…"))
       self.filter_button.clicked.connect(self._on_filter_clicked)  # type: ignore[attr-defined]

       self.start_button = QPushButton(self._tr("qt_button_start", "Start"))
       self.start_button.clicked.connect(self._on_start_clicked)  # type: ignore[attr-defined]

       self.stop_button = QPushButton(self._tr("qt_button_stop", "Stop"))
       self.stop_button.clicked.connect(self._on_stop_clicked)  # type: ignore[attr-defined]

       # Bouton "Analyse" uniquement si un backend est détecté
       self.analysis_button = None
       if self.analysis_backend != "none":
           label = self._tr("qt_button_analyse", "Analyse")
           self.analysis_button = QPushButton(label)
           self.analysis_button.clicked.connect(self._on_analysis_clicked)  # type: ignore[attr-defined]

           # Optionnel : tooltip indiquant le backend choisi
           if self.analysis_backend == "zeanalyser":
               self.analysis_button.setToolTip("Launch ZeAnalyser (quality analysis)")
           elif self.analysis_backend == "beforehand":
               self.analysis_button.setToolTip("Launch Beforehand analysis")
       # Ajout dans la ligne : Filter, Start, Stop, Analyse
       row.addWidget(self.filter_button)
       row.addWidget(self.start_button)
       row.addWidget(self.stop_button)
       if self.analysis_button is not None:
           row.addWidget(self.analysis_button)

       return row
   ```

* **Important :** ne pas changer l’ordre existant des autres boutons, juste rajouter `Analyse` à droite.

#### 5. Gérer le clic sur le bouton « Analyse »

Ajouter une méthode privée dans `ZeMosaicQtMainWindow` :

```python
def _on_analysis_clicked(self) -> None:
    """
    Handler temporaire pour le bouton 'Analyse'.

    Pour l'instant :
      - logge le backend détecté
      - affiche un message informatif à l'utilisateur
    La vraie intégration (lancement auto de ZeAnalyser / Beforehand avec paramètres)
    sera faite plus tard.
    """
    backend = self.analysis_backend
    root = self.analysis_backend_root

    # Sécurité : si plus de backend détecté, désactiver le bouton
    if backend == "none" or root is None:
        if self.analysis_button is not None:
            self.analysis_button.setEnabled(False)
        self._append_log("[INFO] [Analyse] No analysis backend available anymore.")
        QMessageBox.information(
            self,
            "Analysis",
            "No analysis backend is available. Please check your installation.",
        )
        return

    # Message selon le backend
    if backend == "zeanalyser":
        title = "ZeAnalyser detected"
        msg = (
            f"ZeAnalyser installation detected here:\n\n{root}\n\n"
            "The integration is not wired yet.\n"
            "You can launch ZeAnalyser manually from this folder for now."
        )
    else:
        title = "Beforehand analysis detected"
        msg = (
            f"'beforehand' analysis workflow detected here:\n\n{root}\n\n"
            "The integration is not wired yet.\n"
            "You can run your Beforehand tools manually from this folder."
        )

    self._append_log(f"[INFO] [Analyse] Backend={backend}, root={root}")
    QMessageBox.information(self, title, msg)
```

* `_append_log` existe déjà dans cette classe (utilisé pour la zone de log en bas).
* Ce handler ne modifie **aucun** comportement existant du pipeline : il ne lance pour l’instant que des messages.

> **Phase 2 ultérieure (hors scope de cette mission)** : remplacer le contenu de `_on_analysis_clicked` par un véritable lancement de ZeAnalyser / Beforehand (subprocess, arguments CLI, etc.).

#### 6. Internationalisation (optionnel mais propre)

Dans `zemosaic_localization.py` :

* Ajouter une entrée pour la clé `qt_button_analyse` dans les dictionnaires EN/FR :

  * EN : `"qt_button_analyse": "Analyse"`
  * FR : `"qt_button_analyse": "Analyse"`

Si le fichier possède déjà une convention spécifique pour les boutons, s’y conformer (ordre, commentaires, etc.).

---

### Contraintes

* **Ne pas toucher** :

  * `zemosaic_gui.py` (GUI Tk).
  * `zemosaic_worker.py`, `grid_mode.py`, `zemosaic_stack_core.py`, etc.
  * La logique de stacking / grid / GPU / multithread déjà en place.
* Pas de nouvelle dépendance externe.
* La fonctionnalité doit rester **gracieuse** en cas d’erreur :
  → en cas d’exception lors de la détection, on retombe sur `backend="none"` et le bouton n’apparaît simplement pas.

---

### Tests attendus

1. **Cas 1 : aucun backend**

   * Arborescence sans `zeanalyser` ni `seestar/beforehand`.
   * Lancer ZeMosaic Qt.
   * Vérifier que :

     * Aucun bouton « Analyse » n’est visible dans la barre `Filter / Start / Stop`.
     * Aucun log lié à l’analyse n’apparaît.

2. **Cas 2 : uniquement Beforehand**

   * Créer un dossier `seestar/beforehand` à côté de `zemosaic`.
   * Lancer ZeMosaic Qt.
   * Vérifier que :

     * Un bouton « Analyse » apparaît.
     * Le tooltip mentionne Beforehand.
     * Clic sur le bouton → message `Beforehand` + log `[Analyse] Backend=beforehand`.

3. **Cas 3 : ZeAnalyser + Beforehand**

   * Créer `zeanalyser` **et** `seestar/beforehand`.
   * Lancer ZeMosaic Qt.
   * Vérifier que :

     * Le bouton « Analyse » est présent.
     * Le tooltip mentionne ZeAnalyser.
     * Clic → message ZeAnalyser + log `Backend=zeanalyser`.

4. **Cas 4 : backend supprimé en cours de route (edge case)**

   * Lancer ZeMosaic avec `zeanalyser` présent.
   * Supprimer (ou renommer) le dossier `zeanalyser` pendant que la GUI tourne.
   * Cliquer sur « Analyse » :

     * Le code doit gérer le cas proprement : soit message “no backend available anymore”, soit bouton désactivé.

---

### Hors scope

* Intégration directe avec ZeAnalyser (CLI, API, etc.).
* Passage automatique des chemins d’input/output à ZeAnalyser / Beforehand.
* Modifications du worker, du pipeline, ou du grid mode.
* Portage de la fonctionnalité vers le GUI Tk.

