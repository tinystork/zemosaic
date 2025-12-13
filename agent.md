# 📄 `agent.md` (version corrigée et verrouillée)

```markdown
# 🎯 Mission — Diagnostic du décalage vert (mode Classic)
# 🔒 IMPORTANT : réutiliser le système de logging EXISTANT (GUI Qt)

## Contexte clé (à lire AVANT toute modification)
⚠️ Le GUI Qt de ZeMosaic possède DÉJÀ un menu déroulant :
- Section : "Logging / progress"
- Champ : "Logging level"
- Valeurs existantes : Info / Debug (au minimum)

👉 Ce menu existe déjà.
👉 Il fonctionne déjà côté GUI.
👉 IL NE FAUT PAS créer un nouveau système de logging.
👉 IL NE FAUT PAS ajouter un nouveau réglage utilisateur.
👉 IL FAUT UNIQUEMENT PROPAGER la valeur EXISTANTE jusqu’au worker.

---

## Objectif
Identifier précisément **à quelle phase du pipeline Classic**
le canal vert commence à dériver par rapport à R et B.

Pour cela :
1) S’assurer que le **niveau de log sélectionné dans le GUI Qt**
   est réellement appliqué au **logger du worker**
2) Ajouter des logs DEBUG **ultra ciblés** aux frontières critiques
   (P3 → P4 → P5 → export)

Aucun changement algorithmique.
Aucun refactor.
Logs uniquement.

---

## 🚫 Interdictions strictes
- ❌ Ne PAS créer un nouveau menu de logging
- ❌ Ne PAS créer un nouveau flag debug
- ❌ Ne PAS créer un logger parallèle
- ❌ Ne PAS modifier la logique de calcul des images
- ❌ Ne PAS modifier Grid ou SDS

---

## ✅ Ce qui DOIT être fait (et seulement ça)

---

## 1️⃣ Utiliser le dropdown "Logging level" EXISTANT (GUI Qt)

### Fichier : `zemosaic_gui_qt.py`

- Le dropdown **existe déjà**
- Il fournit déjà une valeur logique (`"Info"`, `"Debug"`, etc.)

👉 Action demandée :
- Récupérer la valeur ACTUELLE de ce dropdown
- La transmettre telle quelle au worker
- Sans transformation exotique
- Sans créer de nouvelle option

Par exemple (conceptuellement) :
- `"Info"` → worker log level INFO
- `"Debug"` → worker log level DEBUG

⚠️ Ne pas créer un nouveau champ UI.
⚠️ Ne pas renommer le champ.
⚠️ Ne pas ajouter de nouvelle clé de config utilisateur.

---

## 2️⃣ Appliquer réellement ce niveau de log dans le worker

### Fichier : `zemosaic_worker.py`

Contexte important :
- Le worker peut être lancé dans un process séparé
- Le niveau de log par défaut est actuellement INFO
- Le chemin "classic legacy" ne respecte pas toujours le niveau demandé

👉 Action demandée :
- Lire le **niveau de log transmis par le GUI Qt existant**
- Appliquer ce niveau :
  - au logger `ZeMosaicWorker`
  - et à ses handlers si nécessaire

Ajouter UN log INFO (unique) au démarrage du worker :
```

[LOGCFG] effective_level=DEBUG source=qt_gui_dropdown

```
ou
```

[LOGCFG] effective_level=INFO source=qt_gui_dropdown

```

But :
- Pouvoir prouver que le choix du dropdown GUI est bien effectif côté worker

---

## 3️⃣ Logs DEBUG ciblés par phase (AUCUN autre log)

Ces logs doivent être conditionnés par :
```

if logger.isEnabledFor(logging.DEBUG):

```

### 🔍 Phase 3 / 3.x — Stack des master tiles (baseline saine)

Objectif :
- Confirmer que la couleur est saine AVANT la mosaïque

Ajouter logs DEBUG :
- Avant `stack_core`
- Après `stack_core`
- Après `_poststack_rgb_equalization` (si appelée)

Mesures à logger (1 ligne par point) :
- min / mean / median par canal
- ratio G/R et G/B
- uniquement sur pixels valides

Labels obligatoires :
- `P3_pre_stack_core`
- `P3_post_stack_core`
- `P3_post_poststack_rgb_eq`

---

### 🔥 Phase 4 / 4.x — Assemblage mosaïque (ZONE CRITIQUE #1)

Objectif :
- Détecter si la dérive apparaît lors de la fusion + coverage

Ajouter logs DEBUG :
- Juste AVANT la fusion finale
- Juste APRÈS la fusion finale

Mesures :
1) Stats RGB globales
2) Stats RGB sur pixels valides uniquement
   - valid = coverage > 0
3) Moyenne RGB pondérée par coverage
4) Ratios G/R et G/B pour (2) et (3)

Labels obligatoires :
- `P4_pre_fusion`
- `P4_post_fusion`

---

### 🔥🔥 Phase 5 — Post-processing global (ZONE CRITIQUE #2)

Objectif :
- Identifier une normalisation RGB globale incorrecte (Classic-only)

Ajouter logs DEBUG :
- Avant tout traitement global
- Après chaque étape suspecte :
  - `_apply_final_mosaic_rgb_equalization`
  - normalisation RGB
  - scaling global

Si une égalisation RGB est appliquée :
- Logger explicitement :
  - cibles
  - gains par canal

Labels :
- `P5_pre_global_post`
- `P5_post_<step_name>`

---

### ⚠️ Phase 6–7 — Export / clamp (secondaire)

Ajouter logs DEBUG uniques :
- dtype avant export
- min / max par canal avant clamp
- dtype après conversion

Labels :
- `P6_pre_export`
- `P7_post_export`

---

## 4️⃣ Utilitaire de stats
- Réutiliser `_dbg_rgb_stats` existant
- L’étendre si nécessaire (coverage / mask)
- AUCUN nouvel utilitaire parallèle

---

## 🎯 Critère de succès
Avec **UN SEUL RUN Classic en Debug**, on doit pouvoir dire :
> “La dérive G/R apparaît pour la première fois en phase X, étape Y.”

👉 Le correctif viendra APRÈS, dans une mission séparée.
