
### Objectif du followup

Après le premier passage de Codex :

* Vérifier que la logique de lancement via `subprocess.Popen()` est correctement implémentée.
* S’assurer que le comportement est robuste, qu’aucun crash n’apparaît.
* Ajuster les détails (logs, messages, imports) si nécessaire.

---

### Check-list de vérification

1. **État interne** [x]

   * `self.analysis_backend` et `self.analysis_backend_root` existent dans la classe principale.
   * `_detect_analysis_backend()` est appelé une fois à l’init et son résultat est stocké.

2. **Bouton Analyse** [x]

   * Le bouton est uniquement créé si `analysis_backend != "none"`.
   * Il est connecté à une méthode `self._on_analysis_clicked`.
   * Le reste de la barre de boutons (`Filter`, `Start`, `Stop`) est inchangé.

3. **Méthode `_launch_analysis_backend`** [x]

   * La méthode existe, et :

     * lit bien `self.analysis_backend` et `self.analysis_backend_root`,
     * gère clairement les cas :

       * `backend == "none"` ou `root is None`,
       * `backend == "zeanalyser"` (script défini),
       * `backend == "beforehand"` (info only),
       * backend inconnu (warning).
   * Elle construit un chemin `script = root / "analyse_gui_qt.py"` pour ZeAnalyser.
   * Elle teste `script.is_file()` avant de lancer le process.
   * En cas d’erreur, elle affiche un `QMessageBox.critical`/`warning` **sans lever d’exception non attrapée**.

4. **Lancement du process** [x]

   * Utilise bien :

     ```python
     cmd = [sys.executable, str(script)]
     subprocess.Popen(
         cmd,
         cwd=str(root),
         close_fds=False,
         shell=False,
         creationflags=0,
     )
     ```

   * `sys` et `subprocess` sont correctement importés.

   * Pas de `shell=True`, pas de chemin hardcodé spécifique à Windows.

   * `_append_log` est appelé avec un `try/except` défensif.

5. **Handler de clic** [x]

   * `_on_analysis_clicked` ne contient plus la vieille `QMessageBox` "integration not wired yet".
   * Il se contente d’appeler `self._launch_analysis_backend()`.

6. **Comportement utilisateur** [x]

   * ZeMosaic ne "freeze" pas quand on clique sur Analyse.
   * En cas de script manquant, le message est explicite et ZeMosaic reste utilisable.
   * La fenêtre ZeAnalyser qui s’ouvre est totalement indépendante (redimensionnable, etc.).

---

### Si quelque chose ne va pas…

* **Le bouton Analyse ne lance rien et pas de message :**

  * Vérifier que `_on_analysis_clicked` appelle bien `_launch_analysis_backend`.
  * Ajouter temporairement un log au début de `_launch_analysis_backend` pour confirmer l’appel.

* **Exception dans la console au clic :**

  * Vérifier que tous les imports (`sys`, `subprocess`, `Path`) sont présents.
  * Vérifier que toutes les branches de `_launch_analysis_backend` se terminent par un `return` ou un `Popen` entouré d’un `try/except`.

* **ZeAnalyser ne se lance pas mais aucune erreur :**

  * Vérifier que `script.is_file()` retourne `True` (logger `script`).
  * Vérifier que `sys.executable` pointe vers un Python correct (logger `sys.executable` dans `_append_log` si besoin).
  * Sur Windows, vérifier que l’antivirus ne bloque pas l’exécution d’un nouveau processus Python.
