# followup.md — Détails d’implémentation (patch chirurgical)

## 1) Ajouter l’état & signaux (dans la classe du dialog)
### 1.1 Ajouter un signal stage
À côté de :
- `_async_log_signal = Signal(str, str)`
- `_auto_group_finished_signal = Signal(object, object)`

Ajouter :
- `_auto_group_stage_signal = Signal(str)`

### 1.2 Ajouter des champs d’instance (dans `__init__`)
Initialiser :
- `self._auto_group_stage_text = ""`
- `self._auto_group_started_at = None` (float perf_counter)
- `self._auto_group_elapsed_timer = None` (QTimer | None)

Connecter :
- `self._auto_group_stage_signal.connect(self._handle_auto_group_stage_update)`

## 2) Ajouter 3 petites méthodes utilitaires (no refactor)
### 2.1 `_start_auto_group_elapsed_timer(stage: str)`
- Stocker `self._auto_group_started_at = time.perf_counter()`
- Stocker `self._auto_group_stage_text = stage`
- Créer (si None) un `QTimer(self)` qui tick toutes les 1000 ms
- À chaque tick :
  - si `self._auto_group_running` est False → stop
  - sinon construire : `f"{self._auto_group_stage_text} (elapsed: {secs}s)"`
  - `self._status_label.setText(...)`

### 2.2 `_stop_auto_group_elapsed_timer()`
- Stop timer si existe (sans le détruire forcément)
- Remettre `self._auto_group_started_at = None`

### 2.3 Slot `_handle_auto_group_stage_update(stage: str)`
- Mettre `self._auto_group_stage_text = stage`
- Optionnel : forcer une mise à jour immédiate du status label (sans attendre 1s)

## 3) Intégrer timer + stage dans le flux existant
### 3.1 Dans `_start_master_tile_organisation(...)`
Juste après :
- `self._auto_group_running = True`
- `self._status_label.setText("Preparing master-tile groups…")`

Ajouter :
- `self._start_auto_group_elapsed_timer(self._localizer.get("filter.cluster.running", "Preparing master-tile groups…"))`

Et désactiver les boutons comme aujourd’hui (inchangé).

### 3.2 Dans `_handle_auto_group_empty_selection()`
Avant de quitter :
- `self._stop_auto_group_elapsed_timer()`
- `self._hide_processing_overlay()` (déjà présent)

### 3.3 Dans le `except` qui entoure `thread.start()` (déjà existant)
Ajouter :
- `self._stop_auto_group_elapsed_timer()` (en plus du hide overlay)

## 4) Rendre le background thread “verbeux” et diagnostiquer la lenteur
### 4.1 Dans `_auto_group_background_task(...)`
Ajouter des jalons **avant** les étapes lourdes, via signaux thread-safe :
- `self._auto_group_stage_signal.emit("Clustering frames…")`
- `self._async_log_signal.emit("Stage: clustering connected groups", "INFO")`

Puis après retour de `_compute_auto_groups` :
- `self._auto_group_stage_signal.emit("Post-processing groups…")`

Avant borrow (si activé) :
- `self._auto_group_stage_signal.emit("Borrowing v1…")`

Avant auto-optimiser (si activé) :
- `self._auto_group_stage_signal.emit("Auto-optimiser…")`

Tout à la fin :
- `self._auto_group_stage_signal.emit("Finalizing…")`

Option bonus (très utile) :
- mesurer `t0 = perf_counter()` au début,
- puis émettre `INFO: done in X.Ys` juste avant `_auto_group_finished_signal.emit(...)`.

## 5) Corriger le cross-thread UI access (point critique)
### 5.1 Dans `_compute_auto_groups(...)`
Repérer la ligne :
- `self._append_log(log_text)`

➡️ Remplacer par une alternative thread-safe :
- soit `messages.append(log_text)` (il sera affiché côté UI à la fin),
- soit `messages.append((log_text, "INFO"))` si tu veux forcer le niveau,
- et/ou (si tu veux le voir “en live”) `self._async_log_signal.emit(log_text, "INFO")`.

Important : **ne plus appeler `_append_log`** depuis cette fonction (elle tourne dans le thread).

## 6) Fiabiliser l’arrêt overlay (le “GIF infini”)
### 6.1 Dans `_handle_auto_group_finished(...)`
Actuel :
- il fait des logs, puis `_apply_auto_group_result`, puis `_hide_processing_overlay()`
- MAIS si `_apply_auto_group_result` lève → overlay ne se cache pas.

Patch :
- Wrap global en `try: ... except Exception as exc: ... finally: ...`
- Dans `finally` :
  - `self._auto_group_running = False` (si pas déjà)
  - `self._set_group_buttons_enabled(True)` (si pas déjà)
  - `self._stop_auto_group_elapsed_timer()`
  - `self._hide_processing_overlay()`

Dans `except` :
- `self._append_log(f"Auto-group apply failed: {exc}", level="ERROR")`
- `self._status_label.setText("Auto-organisation failed.")` (via localizer)

## 7) Checklist de tests manuels (GUI)
1) Dataset petit : cliquer Auto-organize
   - overlay apparaît
   - status label devient “Preparing… (elapsed: Ns)”
   - overlay disparaît à la fin
2) Dataset lourd (1289 frames / centaines de groupes) :
   - status label affiche l’étape courante + elapsed
   - activity log reçoit “Stage: …” pendant l’attente (pas seulement à la fin)
3) Forcer une exception (ex: simuler payload vide) :
   - overlay se cache quand même
   - message erreur visible dans l’activity log + status label

## 8) Notes
- Ne pas modifier les algos de clustering/borrow ici : on veut juste rendre la lenteur explicite.
- Si après instrumentation l’étape “Clustering frames…” reste > minutes, on saura que ce n’est pas “le GIF” mais le clustering (ou une entrée pathologique), et on traitera ensuite la perf côté `_CLUSTER_CONNECTED` / borrow.
