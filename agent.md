# agent.md — ZeMosaic Filter Qt: overlay auto-group qui “tourne éternellement”

## Objectif
Dans `zemosaic_filter_gui_qt.py`, fiabiliser et rendre lisible l’étape “Preparing master-tile groups…” :
1) **L’overlay/GIF doit toujours s’arrêter** (même si exception / payload inattendu).
2) Pendant un traitement long, l’utilisateur doit voir **du progrès** (étape courante + temps écoulé) pour savoir si c’est “juste lent” (clustering/borrow/coverage-first) ou un blocage logique.
3) Éviter toute **interaction Qt depuis le thread** de fond (sinon comportements bizarres, freeze, overlay qui reste, etc.).

## Contraintes
- **Modifier uniquement** `zemosaic_filter_gui_qt.py`.
- Patch **surgical / no refactor** : pas de refonte d’API, pas de nouvelle dépendance.
- Ne pas changer l’algorithme de clustering/borrow : on améliore la robustesse UI + instrumentation.
- Conserver l’usage de `self._localizer.get(key, fallback)` (pas besoin d’ajouter de fichier de traduction).

## Travail demandé
### [x] A) Robustesse overlay (stop garanti)
- Entourer `_handle_auto_group_finished(...)` d’un `try/except/finally` :
  - `finally`: stop timer + `self._hide_processing_overlay()` quoi qu’il arrive.
  - `except`: log erreur dans l’activity log + status label clair.

### [x] B) Feedback “vivant” pendant l’auto-group (pas juste un GIF)
- Ajouter un **elapsed timer** (QTimer) qui met à jour `self._status_label` toutes les ~1s :
  - `"{stage} (elapsed: 37s)"`
- Ajouter un **signal stage** thread-safe (ex: `_auto_group_stage_signal = Signal(str)`) :
  - Le thread de fond émet “Clustering…”, “Coverage-first…”, “Borrowing…”, “Auto-optimiser…”
  - Le UI thread reçoit et met à jour le `stage` utilisé par le timer.

### [x] C) Logs/progress thread-safe
- Dans le thread de fond, émettre quelques logs “progress” via `_async_log_signal.emit(msg, level)` (INFO).
- Supprimer/adapter toute écriture directe dans widgets Qt depuis le thread :
  - Dans `_compute_auto_groups`, remplacer `self._append_log(...)` par :
    - `messages.append(...)` **ou** `_async_log_signal.emit(...)` (mais jamais accès direct UI).

## Critères d’acceptation
- En cas normal : l’overlay disparaît, boutons réactivés, status label final cohérent.
- En cas d’exception : overlay stoppé + message d’erreur visible (pas juste console).
- Sur dataset lourd : status label affiche étape + temps écoulé, activity log reçoit des jalons (on sait si c’est le borrow ou le clustering).
- Pas de warning/erreur Qt de cross-thread access (et pas d’overlay “bloqué”).
