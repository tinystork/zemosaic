
### Mission

Quand l’utilisateur clique sur le bouton **“Analyse”** dans la GUI Qt de ZeMosaic :

* **Lancer automatiquement ZeAnalyser** (si ce backend a été détecté),
* en utilisant **`subprocess.Popen()`**, de façon :

  * **non bloquante** (ZeMosaic continue à tourner),
  * **multi-plateforme** (Windows / Linux / macOS),
  * **robuste** (messages clairs si le script/ Python n’est pas trouvés, aucune exception non gérée).

Le backend **Beforehand** reste pour l’instant non câblé (message explicite seulement).

---

### Contexte

* Fichiers principaux déjà en place :

  * `zemosaic_gui_qt.py` : GUI principale Qt de ZeMosaic.
  * `zemosaic_utils.get_app_base_dir()` : renvoie le dossier `zemosaic` même en mode PyInstaller.
  * `path_helpers.safe_path_isdir` : utilitaire robuste pour tester l’existence d’un dossier.
  * `_detect_analysis_backend()` (déjà créé précédemment) :
    retourne un tuple `(backend, root)` :

    * `backend` ∈ `{"none", "zeanalyser", "beforehand"}`,
    * `root` : `Path` vers le dossier racine du backend, ou `None`.

* La GUI a déjà :

  * un bouton **Analyse** dans `zemosaic_gui_qt.py`,
  * un handler qui, pour l’instant, affiche un `QMessageBox` disant que l’intégration n’est pas encore câblée.

* ZeAnalyser est attendu dans la structure suivante :

  ```text
  .../zeseestarstacker/
      zemosaic/
      zeanalyser/
          analyse_gui_qt.py   <-- script à lancer
  ```

---

### Fichiers à modifier

1. `zemosaic_gui_qt.py`
   (uniquement, ne pas toucher au Tk GUI ni au worker).

---

### Tâches détaillées

#### 1. Vérifier / consolider l’état interne du backend d’analyse

Dans la classe principale de la GUI Qt (par ex. `ZeMosaicQtMainWindow` — utiliser le vrai nom présent dans le fichier) :

* S’assurer qu’il existe deux attributs d’instance :

  ```python
  self.analysis_backend: AnalysisBackend = "none"
  self.analysis_backend_root: Optional[Path] = None
  ```

* Après chargement de la configuration et avant la construction de la barre de boutons, appeler :

  ```python
  self.analysis_backend, self.analysis_backend_root = _detect_analysis_backend()
  ```

* Le bouton **Analyse** doit **seulement** être visible si `analysis_backend != "none"`.

  Exemple dans la méthode qui construit la barre de commandes :

  ```python
  if self.analysis_backend != "none":
      self.analysis_button = QPushButton(self._tr("qt_button_analyse", "Analyse"))
      self.analysis_button.clicked.connect(self._on_analysis_clicked)
      row.addWidget(self.analysis_button)
  else:
      self.analysis_button = None
  ```

> Ne pas changer l’ordre ni le comportement des boutons existants (`Filter`, `Start`, `Stop`).

#### 2. Créer une méthode privée robuste pour lancer le backend

Ajouter dans `zemosaic_gui_qt.py`, dans la classe principale, une nouvelle méthode privée :

```python
def _launch_analysis_backend(self) -> None:
    """
    Launch the selected analysis backend in a separate process (non-blocking).

    - If backend == "zeanalyser": run analyse_gui_qt.py using the current Python
      interpreter (sys.executable).
    - If backend == "beforehand": for now, only show an informational message.
    - If no backend or script is missing: show a warning and return gracefully.
    """
    backend = getattr(self, "analysis_backend", "none")
    root = getattr(self, "analysis_backend_root", None)

    if backend == "none" or root is None:
        QMessageBox.information(
            self,
            "Analysis",
            "No analysis backend is available near this ZeMosaic installation.",
        )
        return

    if backend == "zeanalyser":
        script = root / "analyse_gui_qt.py"
        backend_label = "ZeAnalyser"
    elif backend == "beforehand":
        # For now, we do not auto-launch Beforehand, just inform the user.
        QMessageBox.information(
            self,
            "Beforehand detected",
            f"A 'beforehand' analysis workflow was detected here:\n\n{root}\n\n"
            "Automatic launch is not wired yet. "
            "You can still run your Beforehand tools manually from this folder.",
        )
        return
    else:
        QMessageBox.warning(
            self,
            "Analysis",
            f"Unknown analysis backend: {backend}",
        )
        return

    # At this point we are in the ZeAnalyser case
    if not script.is_file():
        QMessageBox.warning(
            self,
            "Analysis",
            f"Cannot find the analysis script:\n{script}",
        )
        return

    # Use the same Python executable as the running ZeMosaic process
    import sys
    import subprocess

    cmd = [sys.executable, str(script)]

    # Optional: log the command for debugging purposes
    try:
        self._append_log(f"[INFO] [Analysis] Launching {backend_label}: {' '.join(cmd)}")
    except Exception:
        # Never fail just because logging failed
        pass

    try:
        # Non-blocking launch; ZeMosaic stays responsive.
        subprocess.Popen(
            cmd,
            cwd=str(root),
            close_fds=False,  # portable, safe default
            shell=False,      # avoid shell injection issues
            creationflags=0,  # let OS decide; we keep it simple/portable
        )
    except Exception as exc:  # pragma: no cover - defensive
        QMessageBox.critical(
            self,
            "Analysis launch failed",
            f"Failed to launch {backend_label}.\n\n"
            f"Script: {script}\n"
            f"Error: {exc}",
        )
```

Points importants :

* Utiliser **`sys.executable`** pour être sûr de lancer ZeAnalyser avec le **même Python** que ZeMosaic (évite les problèmes de venv).
* `cwd=root` : ZeAnalyser se lance dans son propre dossier (utile s’il dépend de chemins relatifs).
* `shell=False` : plus sûr et plus portable.
* Capturer **toutes** les exceptions et afficher une `QMessageBox` **sans faire crasher ZeMosaic**.
* `_append_log` est appelée dans un `try/except` très large pour ne jamais faire échouer le lancement.

#### 3. Connecter le bouton à cette méthode

Modifier le handler de clic existant pour qu’il appelle simplement `_launch_analysis_backend` :

```python
def _on_analysis_clicked(self) -> None:
    """Slot called when the 'Analyse' button is clicked."""
    self._launch_analysis_backend()
```

Supprimer les anciennes `QMessageBox` “ZeAnalyser detected… The integration is not wired yet.”
Toute la logique est maintenant dans `_launch_analysis_backend`.

#### 4. Imports nécessaires

En haut de `zemosaic_gui_qt.py` :

* Vérifier que `Path` est importé depuis `pathlib` (normalement oui).
* Ajouter si besoin :

  ```python
  import sys
  import subprocess
  ```

> Ne pas ajouter d’autres dépendances.

---

### Contraintes

* **Ne pas modifier :**

  * `zemosaic_gui.py` (GUI Tk),
  * `zemosaic_worker.py`,
  * `grid_mode.py`,
  * ni aucun module scientifique (stacking / alignment / GPU).
* Aucun changement de comportement de pipeline, seulement **un lancement externe d’outil**.
* En cas de problème (pas de backend, script manquant, erreur de lancement),
  **ZeMosaic doit rester stable et utilisable**.

---

### Tests attendus

1. **ZeAnalyser présent, script présent**

   * Arborescence : `…/zeseestarstacker/zemosaic` + `…/zeseestarstacker/zeanalyser/analyse_gui_qt.py` existe.
   * Lancer ZeMosaic Qt.
   * Vérifier :

     * Le bouton **Analyse** est visible.
     * Clic → ouverture d’une nouvelle fenêtre ZeAnalyser.
     * ZeMosaic reste réactif (on peut naviguer, lancer une mosaïque, etc.).
     * Le log affiche un message du type :
       `"[INFO] [Analysis] Launching ZeAnalyser: <cmd>"`.

2. **ZeAnalyser détecté mais script manquant**

   * Supprimer/renommer `analyse_gui_qt.py`.
   * Clic sur **Analyse** :

     * Message `Cannot find the analysis script: ...`.
     * Pas d’exception dans la console.
     * ZeMosaic continue de fonctionner normalement.

3. **Uniquement Beforehand détecté**

   * Pas de `zeanalyser`, mais présence de `seestar/beforehand`.
   * Clic sur **Analyse** :

     * Message d’info expliquant que Beforehand est détecté, mais pas encore câblé.
     * Pas de tentative de `subprocess.Popen()`.

4. **Aucun backend**

   * Aucune des structures attendues n’existe.
   * Le bouton Analyse **ne doit pas apparaître** (ou être désactivé).
   * Aucun message d’erreur au clic (si le bouton est invisible, pas de clic possible).

5. **Multi-plateforme (au moins check basique)**

   * Lancer dans un environnement Linux/Mac si dispo, juste pour vérifier qu’il n’y a :

     * ni `shell=True`,
     * ni chemins Windows hardcodés.

