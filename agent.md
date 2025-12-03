# ZeMosaic – ASTAP concurrency cap (GUI-aware, cpu_count-2 rule)

## 🧭 Contexte

- ZeMosaic utilise ASTAP pour résoudre les WCS, avec une limite de concurrence globale pilotée par :
  - `zemosaic_astrometry.set_astap_max_concurrent_instances(...)` :contentReference[oaicite:0]{index=0}  
  - une valeur de config `astap_max_instances` (persistée dans `zemosaic_config.json`). :contentReference[oaicite:1]{index=1}
- Dans le **GUI Qt principal**, la section *ASTAP configuration* expose un champ :
  - `astap_max_instances` via `_register_spinbox(..., minimum=1, maximum=16)` :contentReference[oaicite:2]{index=2}
- Dans le **Filter GUI Qt**, il existe déjà une logique dynamique de remplissage d’une combo "Max ASTAP instances" :
  - `_populate_astap_instances_combo()` construit la liste `[1..cap]` avec `cap = max(1, os.cpu_count() // 2)` et applique ensuite `set_astap_max_concurrent_instances(...)`. 

L’utilisateur souhaite :
1. Remplacer la limite fixe (8/16) par une limite **dynamique et “safe”** basée sur le nombre de threads CPU.
2. Règle souhaitée : **max_instances = min(cpu_count - 2, 32)**, avec un plancher à 1 (on laisse 2 threads au système).
3. Harmoniser le comportement entre le **GUI principal Qt** et le **Filter GUI Qt**, sans casser la compatibilité existante.

⚠️ Important :  
- Ne pas toucher au pipeline CPU/GPU ni à la logique de stacking / mosaïque.  
- Ne pas introduire de nouvelles dépendances lourdes.

---

## 📂 Fichiers à lire avant toute modification

- `zemosaic_gui_qt.py`  
  - Section ASTAP config, enregistrement du spinbox `astap_max_instances`. 
- `zemosaic_filter_gui_qt.py`  
  - Gestion de l’UI ASTAP instances `_populate_astap_instances_combo`, `_resolve_initial_astap_instances`, `_apply_astap_instances_choice`, `_prepare_astap_configuration`. 
- `zemosaic_astrometry.py`  
  - `set_astap_max_concurrent_instances`, mécanique de sémaphore interne. 
- `zemosaic_config.py`  
  - `DEFAULT_CONFIG["astap_max_instances"]`, `get_astap_max_instances()`. 
- (Optionnel) `en.json` / `fr.json` si tu ajoutes un tooltip explicatif sur la limite. 

---

## 🎯 Objectifs

1. **Introduire une fonction utilitaire unique** qui calcule une limite “recommandée” pour ASTAP en fonction du CPU :  
   - Règle :  
     - `cpu = os.cpu_count() or 2`  
     - `safe = max(1, cpu - 2)` (on laisse 2 threads au système)  
     - `recommended = min(safe, 32)` (cap “hard” à 32 pour éviter les débordements absurdes).
2. **Utiliser cette fonction dans le GUI Qt principal** pour :
   - Fixer dynamiquement le `maximum` du `QSpinBox` `astap_max_instances`.
   - Clamper la valeur persistée / collectée (ne jamais remonter plus que `recommended` aux workers).
3. **Réutiliser la même logique dans le Filter GUI Qt** :
   - Remplacer `cap = max(1, cpu_count // 2)` par l’appel à la même fonction, pour que les deux GUIs soient cohérents.
4. **S’assurer que `set_astap_max_concurrent_instances(...)` reste la seule source de vérité runtime**, appelée depuis les GUIs avec une valeur déjà clampée par la règle `cpu_count - 2`, max 32.
5. **Préserver le comportement existant** :
   - Si un utilisateur a un `astap_max_instances` déjà configuré dans `zemosaic_config.json` :
     - on charge la valeur, on la clamp entre 1 et `recommended`.
     - on met à jour l’UI en conséquence.
   - Si aucune valeur n’est configurée → on peut garder la valeur par défaut (1) ou l’auto-remplacer par `recommended` si tu juges ça plus UX-friendly (voir tâches détaillées ci-dessous).

---

## ✅ Tâches détaillées

### 1. Créer un helper central pour la limite “safe” ASTAP

**Proposition de localisation :** `zemosaic_astrometry.py` (où vit déjà la logique de concurrence ASTAP).

- Ajouter en haut du fichier les imports nécessaires :
  - `import os` si absent.
- Ajouter une fonction :

```python
def compute_astap_recommended_max_instances(
    *,
    reserve_threads: int = 2,
    hard_max: int = 32,
    min_cap: int = 1,
) -> int:
    """
    Compute a 'safe' upper bound for ASTAP concurrency based on CPU count.

    Rule of thumb:
      - leave a few threads for the OS / GUI / Python (reserve_threads)
      - never exceed a conservative hard cap (hard_max)
    """
    try:
        cpu = os.cpu_count() or (reserve_threads + 1)
    except Exception:
        cpu = reserve_threads + 1

    # Leave some room for the OS and other processes
    safe = max(min_cap, cpu - reserve_threads)
    # Apply hard cap to avoid oversubscription on HEDT/servers
    return max(min_cap, min(safe, hard_max))
````

* Exposer cette fonction dans `__all__` si ce pattern est utilisé dans le module (à vérifier).

### 2. Utiliser ce helper dans le GUI Qt principal (`zemosaic_gui_qt.py`)

#### 2.1. Importer le helper

* En haut du fichier, près de l’import de `set_astap_max_concurrent_instances`, ajouter :

```python
from zemosaic_astrometry import (
    set_astap_max_concurrent_instances,
    compute_astap_recommended_max_instances,
)
```

(adapte si le code utilise déjà un `try/except` pour les imports facultatifs).

#### 2.2. Dynamiser la création du spinbox `astap_max_instances`

Dans `_build_solver_tab` (ou la méthode correspondante où tu appelles `_register_spinbox` sur `astap_max_instances`) :

Actuellement :

```python
self._register_spinbox(
    "astap_max_instances",
    astap_layout,
    self._tr("qt_field_astap_max_instances", "Max ASTAP instances"),
    minimum=1,
    maximum=16,
)
```

Remplacer par quelque chose comme :

```python
try:
    astap_cap = compute_astap_recommended_max_instances()
except Exception:
    astap_cap = 16  # fallback conservative

self._register_spinbox(
    "astap_max_instances",
    astap_layout,
    self._tr("qt_field_astap_max_instances", "Max ASTAP instances"),
    minimum=1,
    maximum=astap_cap,
)
```

Optionnel : tu peux aussi ajouter un tooltip sur le widget (`QSpinBox`) pour expliquer la règle (CPU threads - 2, max 32).

#### 2.3. Clamper la valeur de config sur la limite recommandée

Dans `_resolve_astap_max_instances` :

Actuellement :

```python
def _resolve_astap_max_instances(self) -> int:
    try:
        value = int(self.config.get("astap_max_instances", 1) or 1)
    except Exception:
        value = 1
    return max(1, value)
```

Remplacer par :

```python
def _resolve_astap_max_instances(self) -> int:
    try:
        raw = int(self.config.get("astap_max_instances", 1) or 1)
    except Exception:
        raw = 1
    parsed = max(1, raw)
    try:
        cap = compute_astap_recommended_max_instances()
    except Exception:
        cap = parsed  # no extra clamp if helper fails
    return max(1, min(parsed, cap))
```

* Optionnel mais recommandé : après avoir chargé la config et initialisé les widgets, si la valeur clamped diffère de la valeur brute, mettre à jour le spinbox via `_update_widget_from_config` pour refléter visuellement le clamp.

#### 2.4. Conserver et utiliser `_apply_astap_concurrency_setting`

Ne pas modifier la signature, mais vérifier que l’appel continue d’utiliser la valeur déjà clampée :

```python
def _apply_astap_concurrency_setting(self) -> None:
    instances = self._resolve_astap_max_instances()
    os.environ["ZEMOSAIC_ASTAP_MAX_PROCS"] = str(instances)
    if set_astap_max_concurrent_instances is not None:
        try:
            set_astap_max_concurrent_instances(instances)
        except Exception:
            pass
```

La seule différence est que `_resolve_astap_max_instances` ne pourra plus renvoyer une valeur supérieure à `compute_astap_recommended_max_instances()`.

### 3. Harmoniser le Filter GUI Qt (`zemosaic_filter_gui_qt.py`)

#### 3.1. Importer le helper

* En haut du fichier, à côté des imports ASTAP existants (où `set_astap_max_concurrent_instances` est importé), ajouter :

```python
from zemosaic_astrometry import compute_astap_recommended_max_instances
```

(avec le même pattern `try/except` que pour les autres imports optionnels si nécessaire).

#### 3.2. Remplacer la logique de cap dans `_populate_astap_instances_combo`

Actuellement :

```python
cpu_count = os.cpu_count() or 2
cap = max(1, cpu_count // 2)
options = {str(i): i for i in range(1, cap + 1)}
```

Remplacer par :

```python
try:
    cap = compute_astap_recommended_max_instances()
except Exception:
    cpu_count = os.cpu_count() or 2
    cap = max(1, cpu_count // 2)  # fallback actuel

options = {str(i): i for i in range(1, cap + 1)}
```

Ainsi :

* Le Filter GUI et le Main GUI partagent la même règle de limite.
* En cas d’échec du helper (import, erreur inattendue), on garde le comportement actuel (`cpu_count // 2`).

#### 3.3. Conserver le warning multi-instance déjà présent

Ne touche pas à `_apply_astap_instances_choice` et au warning utilisateur (message “Access violation popup” etc.). 
Ce warning doit continuer à s’afficher dès que l’utilisateur dépasse `1` instance, même si la limite max est désormais plus élevée.

### 4. (Optionnel) Ajuster `DEFAULT_CONFIG["astap_max_instances"]`

Dans `zemosaic_config.py`, la valeur par défaut est actuellement :

```python
"astap_max_instances": 1,
```

Tu peux soit :

* **A.** La laisser à 1 (comportement plus conservateur par défaut, l’utilisateur monte ensuite la valeur dans le GUI).
* **B.** L’augmenter à quelque chose comme 4, en sachant qu’elle sera clampée par `compute_astap_recommended_max_instances()`.

**Ne change pas** la signature de `get_astap_max_instances()` ; assure-toi juste qu’elle ne renvoie jamais moins de 1 et laisse le clamp final au niveau des GUIs + runtime setter.

---

## 🔍 Tests / validations attendus

### Tests unitaires / rapides

* Ajouter un petit test (ou au minimum un bloc de debug manuel) pour `compute_astap_recommended_max_instances()` avec différents mocks de `os.cpu_count()` :

  * cpu=4 → recommended=2 (4-2=2)
  * cpu=8 → recommended=6
  * cpu=16 → recommended=14 (clampé à 14, < 32)
  * cpu=64 → safe=62, recommended=32 (clamp hard).

### Tests manuels (GUI)

1. Sur une machine de dev :

   * Lancer `python zemosaic_gui_qt.py`.
   * Aller dans l’onglet/section **ASTAP configuration**.
   * Vérifier que le spinbox “Max ASTAP instances” a pour maximum :

     * `min(os.cpu_count() - 2, 32)`.
2. Modifier la valeur dans le GUI (ex.: mettre le maximum).

   * Fermer puis relancer le GUI.
   * Vérifier que la valeur affichée après rechargement ne dépasse pas la limite recommandée.
3. Lancer un run avec plusieurs tuiles nécessitant ASTAP :

   * Vérifier dans les logs que `set_astap_max_concurrent_instances` est bien appelée avec la valeur choisie.
4. Ouvrir le **Filter GUI Qt** :

   * Vérifier que la combo “Max ASTAP instances” propose les mêmes bornes que le spinbox du main GUI (1 → `recommended`).
   * Monter à une valeur >1, vérifier que le warning multi-instance s’affiche toujours.

---

## 🧱 Contraintes / garde-fous

* Ne pas modifier :

  * La logique de résolution ASTAP elle-même (commande, options, retries, etc.).
  * Le comportement CPU/GPU du pipeline de stacking ou Phase 5.
* Ne pas introduire de nouvelles dépendances (psutil, numpy, etc.) dans des modules qui n’en avaient pas besoin pour cette fonctionnalité.
* Respecter le style existant (nommage, logging, type hints) pour garder le code lisible et cohérent.

