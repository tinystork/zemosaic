# ZeMosaic — ROADMAP UNIFIEE FULL (zero perte d'information)

Date: 2026-03-13
But: conserver la lisibilite de la version unifiee ET preserver le detail complet des versions sources.

---

## Statut de ce document (règle d'interprétation)

- **Normatif (fait foi pour l'exécution):** section **A. Canon executable (version unifiee courte)**.
- **Référence (non normative):** sections **B** et **C** (appendices de conservation).
- En cas de divergence de formulation, **la section A prime**.

## A. Canon executable (version unifiee courte)

# ZeMosaic — ROADMAP UNIFIÉE (Idiot-Proof)
## Retrait Tkinter legacy + bascule Qt-only officielle

Date: 2026-03-13  
Owner: Tristan / ZeMosaic core  
Version: Unified v1 (fusion de `ROADMAP_REMOVE_TKINTER.md` + `ROADMAP_REMOVE_TKINTER_V4.md`)

---

## 0) Résumé exécutif (1 minute)

Cette roadmap distingue **3 sujets**:

1. **Qt-only officiel (P0)**: frontend officiel ZeMosaic doit être 100% Qt.
2. **Tk-free sur chemins officiels/headless validés (P0)**: pas d’import Tk latent via config/worker.
3. **Sort des outils annexes Tk (P2)**: ex. `lecropper.py`, à traiter après stabilisation.

Règle d’or:
> D’abord rendre Tk inutile sur le runtime officiel, ensuite retirer le legacy, puis décider des annexes.

---

## 1) Décisions d’architecture (verrouillées)

### 1.1 Objectifs
- **A (obligatoire)**: Qt-only pour l’application officielle.
- **B (obligatoire)**: zéro dépendance Tk sur chemins officiels + headless validés.
- **C (optionnel après release)**: purge totale Tk du repo.

### 1.2 Décision explicite sur `lecropper`
- `lecropper.py` est classé **outil annexe** (pas frontend officiel).
- **Bloquant P0**: le runtime officiel ne doit plus dépendre de `lecropper`.
- Portage Qt de `lecropper` = **phase séparée** (après release Qt-only), si stratégique.

---

## 2) Scope (sans ambiguïté)

## 2.1 In-scope P0 (obligatoire)
- `run_zemosaic.py`
- `zemosaic_gui_qt.py`
- `zemosaic_filter_gui_qt.py`
- `zemosaic_config.py` (import-safe sans Tk)
- messages d’erreur de démarrage
- packaging/docs du frontend officiel
- migration config legacy
- parcours headless validés (définis en S0)

## 2.2 Out-of-scope première vague
- portage Qt complet de `lecropper`
- suppression totale de Tk dans tout le repo
- refactor esthétique large hors migration

## 2.3 Scope secondaire (post-release)
- `zemosaic_gui.py`
- `zemosaic_filter_gui.py`
- `lecropper.py`
- `zequalityMT.py`
- wrappers/helpers Tk annexes

---

## 3) Source de vérité unique (anti-redondance)

- Parité fonctionnelle Qt → **S1**
- Runtime Qt-only sans fallback Tk → **S2**
- Migration config idempotente → **S3**
- Packaging/docs/release notes → **S4**
- Validation QA + CI no-Tk → **S5**
- Sort final des annexes (`lecropper`) → **S6**

Toute autre section ne fait que référencer ces sections canoniques.

---

## 4) Plan d’exécution (ordre impératif)

**S0 → S1 → S2 → S3 → S4 → S5**, puis éventuellement **S6**.

Interdit:
- lancer le portage Qt de `lecropper` pendant S1/S2 si ça retarde la bascule officielle.
- supprimer massivement sans audit S0.

---

## S0 — Audit verrouillé et bornage
**Durée**: 2–3 jours | **Priorité**: P0

### Livrables
1. Inventaire des usages Tk (`tkinter`, `messagebox`, `filedialog`, `ttk`, `Tk()`, transitifs).
2. Classification par fichier:
   - official runtime
   - official headless
   - legacy GUI
   - standalone utility
   - build/doc/test only
3. Parcours headless **officiellement validés** (liste fermée).
4. Stratégie `zemosaic_config.py`:
   - Tk-free direct, ou
   - split core/legacy, ou
   - lazy import strict hors officiel.
5. Statut initial de `lecropper` (supported/internal/legacy/deprecated).
6. Matrice parité Qt initiale (OK / gap / bloquant / hors-scope).

### Done S0
- scope P0 validé écritement
- headless borné
- stratégie config décidée
- statut `lecropper` décidé

---

## S1 — Parité fonctionnelle Qt stricte
**Durée**: 4–6 jours | **Priorité**: P0

### Objectif
Ne laisser **aucun workflow officiel critique** dépendre de Tk.

### Travail
- fermer les gaps UI/UX Qt
- parité persistance config
- retirer discours “Qt preview / Tk stable”
- traiter features backend encore vivantes mais cachées en Qt:
  - exposer, ou
  - classer hors-scope/legacy explicitement

### Done S1
- matrice parité à jour
- aucun blocant P0/P1 officiel

---

## S2 — Cutover runtime Qt-only
**Durée**: 2–3 jours | **Priorité**: P0

### Travail
1. Supprimer fallback Tk dans `run_zemosaic.py`.
2. Supprimer message boxes Tk de démarrage.
3. Rendre `zemosaic_config.py` import-safe sans Tk.
4. Basculer défaut config backend sur Qt.
5. Retirer choix de backend dans UI Qt.
6. Durcir messages dépendance (PySide6 requis pour frontend officiel).
7. Découpler `lecropper` du runtime officiel (zéro import direct/indirect).

### Gate S2 (obligatoire)
Environnement sans Tk:
- lancement officiel sans fallback Tk
- `import zemosaic_config` OK
- `import zemosaic_worker` sur chemins headless validés OK
- absence de `lecropper` ne casse pas l’officiel

---

## S3 — Migration config + nettoyage officiel
**Durée**: 3–5 jours | **Priorité**: P0

### Travail
- supprimer branches mortes backend Tk/Qt dans officiel
- migrer config legacy (`preferred_gui_backend=tk` -> `qt`)
- neutraliser clés legacy inutiles
- garantir **idempotence** load/save (round-trip)

### Done S3
- migration stable en 1 lancement
- round-trip idempotent
- runtime officiel sans références Tk actives

---

## S4 — Packaging / docs / release notes
**Durée**: 2–3 jours | **Priorité**: P1

### Travail
- nettoyer spec/scripts build
- vérifier artefacts finaux (pas seulement scripts)
- docs user + docs dev alignées
- release notes breaking change claires
- statut public des annexes (dont `lecropper`) explicité

### Done S4
- build propre
- docs synchronisées
- release notes prêtes

---

## S5 — Validation finale (QA + CI)
**Durée**: 2 jours | **Priorité**: P0

### Travail
- smoke tests multi-OS
- tests migration config (fixtures legacy multi-plateformes)
- tests dépendances manquantes (PySide6 absent, erreurs lancement)
- tests headless bornés
- CI `no-Tk-on-official-path`

### Done S5
- QA sign-off
- CI verte
- aucun fallback/import Tk latent sur chemins officiels validés

---

## S6 — Annexes Tk / sort final de `lecropper` (post-release)
**Durée**: 2–6 jours | **Priorité**: P2

### Options `lecropper`
- **A** legacy standalone gelé (non packagé officiel)
- **B** portage Qt dédié (préféré si stratégique)
- **C** dépréciation/retrait

### Done S6
- décision explicite + owner + échéance
- isolement réel (si legacy) ou parité mini + tests (si portage)

---

## 5) Checklist GO/NO-GO release Qt-only (bloquante)

**GO seulement si 7/7:**
1. runtime officiel sans fallback Tk
2. `zemosaic_config.py` importable sans Tk
3. chemins headless validés sans Tk
4. aucune dépendance runtime officielle à `lecropper`/annexes Tk
5. migration config idempotente validée
6. docs + release notes alignées
7. CI `no-Tk-on-official-path` verte

Sinon: **NO-GO**.

---

## 6) Risques majeurs et garde-fous

- **Scope creep** → short-list P0 figée (ci-dessus)
- **Redondance des exigences** → section canonique unique par sujet
- **Faux Qt-only** (Tk via config/erreurs) → gates S2/S5 obligatoires
- **Migration config cassée** → fixtures + round-trip idempotent
- **Exceptions permanentes** → registre exceptions (owner + date)
- **Packaging trompeur** → audit artefacts build finaux

---

## 7) 3 artefacts de pilotage obligatoires

1. **Checklist P0 Qt-only** (versionnée)
2. **Registre exceptions temporaires** (owner, échéance, justification)
3. **Matrice tests officiels** (GUI + headless bornés)

---

## 8) Instructions “agent-proof”

- Work only on the next unchecked item.
- Keep changes surgical.
- Do not refactor unrelated code.
- Prove runtime official path has no Tk dependency.
- Prove `zemosaic_config` import safety without Tk.
- If `memory.md` / `agent.md` / `followup.md` absent, log in `MIGRATION_LOG.md`.
- Always state in-scope vs out-of-scope in each PR/report.

---

## 9) Verdict final

Le plan unifié est volontairement strict mais exécutable:
- il sécurise le cutover Qt-only,
- évite les faux positifs “migration terminée”,
- et garde `lecropper` sous contrôle sans bloquer la release officielle.

---

## B. Appendice de conservation — contenu detaille source 1

> Source: ROADMAP_REMOVE_TKINTER.md

# ZeMosaic — Roadmap durci : retrait du GUI Tkinter legacy

Date: 2026-03-13  
Owner: Tristan / ZeMosaic core  
Statut: proposition durcie, prête à servir de base d’exécution  

---

## 1) Décision d’architecture à graver noir sur blanc

Le point clé est de **séparer deux objectifs qui ne sont pas équivalents** :

1. **Objectif A — Qt-only pour l’application officielle ZeMosaic**  
   Cela couvre le lanceur, la fenêtre principale, la fenêtre de filtre, la config, les messages d’erreur de démarrage, le packaging et la documentation.

2. **Objectif B — purge complète de Tkinter de tout le repository**  
   Cela inclut aussi les outils annexes / standalone qui peuvent encore embarquer Tk (`lecropper.py`, `zequalityMT.py`, etc.).

Le présent roadmap recommande de traiter **A comme scope primaire obligatoire**, puis de traiter **B comme une phase optionnelle explicite** une fois la release Qt-only stabilisée.

Cette séparation évite deux erreurs classiques :
- supprimer trop tôt des utilitaires encore utiles ;
- laisser croire que “Qt-only” est terminé alors qu’il reste du Tk sur le chemin officiel de lancement.

---

## 2) Scope retenu

### Scope primaire (obligatoire)
Le chemin officiel ZeMosaic doit devenir **100% Qt** pour :
- `run_zemosaic.py`
- `zemosaic_gui_qt.py`
- `zemosaic_filter_gui_qt.py`
- `zemosaic_config.py` (**à rendre import-safe sans Tk sur tous les chemins officiels et headless**)
- scripts/builds/docs liés au lancement officiel
- migration de configuration utilisateur

### Scope secondaire (optionnel, à décider explicitement)
Éléments encore basés sur Tk mais pouvant être considérés comme outils annexes / standalone :
- `zemosaic_gui.py` si encore présent dans le repo réel
- `zemosaic_filter_gui.py`
- `lecropper.py`
- `zequalityMT.py`
- tout helper ou wrapper Tk historique hors chemin officiel

### Hors-scope par défaut pour la première vague
Sauf décision contraire explicite, **ne pas casser ni refactorer agressivement** les outils annexes standalone tant que l’application officielle Qt-only n’est pas verrouillée.

---

## 3) Constat de départ dans le code actuel

Le roadmap doit partir de la réalité actuelle du code, pas d’une intention abstraite.

### Dépendances Tk encore visibles sur le chemin officiel
- `run_zemosaic.py` importe encore `tkinter` et `messagebox`.
- `run_zemosaic.py` affiche encore des boîtes d’erreur/warning via Tk.
- `run_zemosaic.py` garde encore une logique de fallback vers Tk quand Qt n’est pas disponible.
- `zemosaic_config.py` a encore `preferred_gui_backend = "tk"` par défaut.
- `zemosaic_config.py` doit être considéré comme un point critique car il peut être importé depuis le runtime Qt et depuis des chemins non-GUI/headless.
- `zemosaic_gui_qt.py` expose encore un choix de backend Tk/Qt dans l’UI.
- `zemosaic_gui_qt.py` contient encore le discours “Qt preview / Tk stable”.
- `zemosaic_filter_gui_qt.py` contient encore un message indiquant “install PySide6 or use the Tk interface instead”.

### Dépendances Tk hors chemin officiel mais présentes dans le repo
- `zemosaic_filter_gui.py`
- `lecropper.py`
- `zequalityMT.py`

Conséquence : **la suppression de Tk ne peut pas être traitée comme une simple suppression de fichiers**. C’est une migration runtime, UX, config, packaging et communication.

---

## 4) Principe directeur

La bonne définition de “Qt-only” est la suivante :

> Un utilisateur qui lance ZeMosaic via le chemin officiel ne doit dépendre de Tk ni en cas nominal, ni en cas d’erreur de démarrage, ni dans sa config, ni dans l’interface, ni dans la documentation.

Tant que ce n’est pas vrai, le chantier n’est pas terminé.


Corollaire important :

> “Qt-only” ne signifie pas seulement “le launcher ouvre une fenêtre Qt”.  
> Cela signifie aussi que les modules communs (`zemosaic_config.py`, worker/config round-trip, imports headless) ne doivent plus réintroduire Tk par effet de bord.


---

## 5) Stratégie recommandée

Ordre impératif :  
**S0 → S1 → S2 → S3 → S4 → S5**  
Puis éventuellement **S6** pour la purge complète du repo.

**Ne pas accélérer S3 avant validation complète de S1 et S2.**  
Supprimer tôt est facile. Réparer une régression utilisateur après coup est pénible.

---

## S0 — Audit verrouillé et bornage du scope

**Durée estimée:** 2–3 jours  
**Effort estimé:** 2–4 JH  
**Priorité:** P0

### Objectifs
- Geler le périmètre exact.
- Produire un inventaire fiable des dépendances Tk.
- Classer les usages Tk par criticité.

### Tickets
1. **Inventaire exhaustif des imports et usages Tk**
   - Rechercher `tkinter`, `messagebox`, `filedialog`, `ttk`, `Tk()` et wrappers associés.
   - Produire une liste par fichier avec type d’usage :
     - `official runtime path`
     - `legacy GUI`
     - `standalone utility`
     - `build/doc/test only`

2. **Décision formelle de scope**
   - Trancher par écrit si `lecropper.py` et `zequalityMT.py` font partie de la purge immédiate ou non.
   - Trancher par écrit si `zemosaic_filter_gui.py` doit survivre temporairement comme legacy tool ou être supprimé dans la première vague.

3. **Audit du chemin de lancement officiel**
   - Vérifier le comportement réel de `run_zemosaic.py` en cas :
     - Qt disponible
     - Qt indisponible
     - erreur d’import worker
     - erreur de lancement GUI
   - Identifier tout chemin qui réimporte Tk.

4. **Audit critique de `zemosaic_config.py`**
   - Identifier tous les imports Tk directs ou indirects dans `zemosaic_config.py`.
   - Décider si le module doit être :
     - rendu totalement Tk-free,
     - scindé entre “core config pur” et helpers GUI legacy,
     - ou encapsulé derrière un import lazy strictement hors chemin officiel.
   - Exiger un résultat clair : `zemosaic_config.py` doit être importable sans Tk sur tous les chemins officiels et headless.

5. **Matrice de parité UI / UX Qt**
   - Lister les écrans, réglages, dialogues, logs, raccourcis, options de persistance et comportements attendus.
   - Marquer pour chaque item : `OK`, `à compléter`, `bloquant suppression Tk`, `hors-scope`.

6. **Audit des features backend encore vivantes mais cachées en Qt**
   - Repérer les fonctionnalités encore implémentées côté worker/backend mais non exposées, masquées ou “temporarily hidden” côté Qt.
   - Classer chaque cas :
     - `gap à fermer avant retrait Tk`
     - `hors-scope assumé`
     - `legacy volontairement non reconduit`
   - Éviter qu’une feature disparaisse “sans bruit” au prétexte que l’UI Qt ne l’expose pas encore.

7. **Audit packaging / docs**
   - Relever toutes les mentions “Tk stable / Qt preview / fallback Tk”.
   - Vérifier scripts de build, spec, dépendances et messages utilisateur.

### Critères de done
- Scope primaire validé par écrit.
- Liste exhaustive des usages Tk, catégorisée.
- Tableau des blocants P0/P1 avant suppression.
- Décision explicite sur le sort des outils annexes.
- Décision explicite sur la stratégie `zemosaic_config.py` (Tk-free / split / lazy import hors scope officiel).
- Liste des features backend vivantes mais non exposées en Qt, avec décision documentée.

### Interdits
- Pas de suppression de code Tk pendant S0.
- Pas de refactor opportuniste hors sujet.

---

## S1 — Parité fonctionnelle Qt stricte

**Durée estimée:** 4–6 jours  
**Effort estimé:** 4–8 JH  
**Priorité:** P0

### Objectifs
- Garantir qu’aucun workflow critique officiel n’exige encore Tk.
- Éliminer l’idée même de “Qt preview” côté produit.

### Tickets
1. **Fermer les gaps fonctionnels Qt**
   - Réglages principaux
   - dialogues critiques
   - logs / feedback utilisateur
   - callbacks de démarrage / arrêt
   - comportements liés au filtre si celui-ci fait partie du frontend officiel

2. **Parité de persistance config**
   - Vérifier que tous les réglages utiles sauvegardés/relus en pratique le sont correctement depuis Qt.
   - Supprimer les ambiguïtés de comportement entre interface et config persistée.

3. **Nettoyage UX préparatoire**
   - Retirer le vocabulaire “preview” si Qt est le frontend cible.
   - Préparer la disparition du choix de backend dans l’UI.

4. **Traitement des features backend vivantes mais cachées en Qt**
   - Pour chaque feature encore présente côté backend/worker mais absente ou masquée côté Qt :
     - décider si elle doit être exposée avant retrait Tk,
     - ou déclarée explicitement hors-scope / legacy.
   - Documenter noir sur blanc toute non-parité assumée.

5. **Validation workflows critiques**
   - Démarrage GUI
   - chargement/sauvegarde config
   - lancement pipeline standard
   - Grid mode
   - SDS si concerné par le frontend officiel
   - ouverture des outils officiellement exposés depuis l’UI
   - fermeture propre / reprise / logs / export de sortie

### Critères de done
- Matrice de parité complétée.
- Aucun blocant P0/P1 restant pour l’utilisateur officiel.
- Toute feature backend non exposée en Qt est soit fermée, soit explicitement classée hors-scope/legacy.
- Qt n’est plus présenté comme expérimental dans le produit.

### Interdits
- Ne pas encore supprimer le fallback runtime tant que la parité n’est pas prouvée.

---

## S2 — Qt-only au runtime officiel

**Durée estimée:** 2–3 jours  
**Effort estimé:** 2–4 JH  
**Priorité:** P0

### Objectifs
- Faire de Qt le **seul** backend supporté pour le lancement officiel.
- Faire disparaître toute dépendance Tk sur le chemin nominal et sur le chemin d’erreur de lancement.

### Tickets
1. **Supprimer le fallback backend dans `run_zemosaic.py`**
   - Plus de “si Qt indisponible → bascule Tk”.
   - Plus de `--tk-gui` sur le chemin officiel.
   - Plus de sélection implicite/explicite Tk via variable d’environnement pour l’app officielle.

2. **Supprimer les message boxes Tk de démarrage**
   - Toute erreur de lancement doit être gérée en console, Qt, ou via un mécanisme neutre.
   - Aucune création de root Tk pour signaler une erreur critique.

3. **Rendre `zemosaic_config.py` import-safe sans Tk**
   - Le module doit être importable sans Tk sur le runtime officiel et sur les chemins headless/non-GUI.
   - Si nécessaire, séparer la logique “core config” des helpers GUI legacy.
   - Interdire qu’un import innocent de config réintroduise Tk par effet de bord.

4. **Basculer la config officielle sur Qt**
   - Valeur par défaut = `qt`.
   - Lire encore les anciennes configs le temps de la migration.
   - Préparer la disparition de `preferred_gui_backend` comme réglage exposé à l’utilisateur.

5. **Supprimer le choix de backend dans l’UI Qt**
   - Enlever le groupe “Preferred GUI backend”.
   - Enlever toute copie “Classic Tk GUI (stable)” / “Qt GUI (preview)”.

6. **Durcir les messages d’absence de dépendance Qt**
   - Message clair : ZeMosaic requiert PySide6 pour son frontend officiel.
   - Ne plus suggérer “use the Tk interface instead” dans les modules officiels.

7. **Entrypoint officiel explicite (anti-ambiguïté)**
   - L’entrypoint officiel doit cesser d’importer `zemosaic_gui.py` et importer directement `zemosaic_gui_qt.py`.
   - Aucune logique officielle ne doit conserver un chemin implicite vers `zemosaic_gui.py`.

8. **Règle de wording produit après S2 (anti-faux-positifs)**
   - Une fois S2 terminé, aucun module officiel ne doit encore se présenter comme “optionnel” ou “coexistant” avec Tk.
   - Supprimer les formulations du type “Qt preview / Tk stable”, “Preferred GUI backend”, “use Tk interface instead”,
     ou toute mention laissant entendre que Tk reste une alternative officielle.

### Critères de done
- L’application officielle démarre uniquement en Qt.
- Aucun import ni appel Tk sur le chemin officiel nominal.
- Aucun import ni appel Tk sur le chemin d’erreur de démarrage officiel.
- `zemosaic_config.py` est importable sans Tk sur les chemins officiels et headless.
- L’utilisateur ne peut plus choisir Tk depuis l’UI officielle.
- L’entrypoint officiel importe directement `zemosaic_gui_qt.py` (et n’importe plus `zemosaic_gui.py`).
- Aucun module officiel ne se décrit encore comme coexistant avec Tk après S2.

### Gate de sortie obligatoire
Tester explicitement l’environnement **sans Tk disponible** et vérifier :
- que le lancement officiel n’en dépend plus ;
- que `import zemosaic_config` ne réintroduit pas Tk ;
- qu’aucun chemin officiel ou headless n’échoue à cause d’un import Tk latent.

---

## S3 — Nettoyage legacy dans le codebase officiel

**Durée estimée:** 3–5 jours  
**Effort estimé:** 4–7 JH  
**Priorité:** P0

### Objectifs
- Retirer la dette technique Tk devenue morte dans le périmètre officiel.
- Garder une migration utilisateur propre.

### Tickets
1. **Supprimer les branches mortes liées au backend**
   - Conditions, flags, commentaires, docstrings et code mort autour de la coexistence Tk/Qt.

2. **Migration de configuration legacy**
   - Si une config contient `preferred_gui_backend = tk`, la rewriter vers `qt`.
   - Si `preferred_gui_backend_explicit` n’a plus de sens, prévoir sa neutralisation puis sa suppression.
   - La migration doit être silencieuse, robuste et testée.

3. **Nettoyage des modules legacy du frontend officiel**
   - Supprimer du packaging et du runtime officiel ce qui n’est plus censé être callable.
   - Si `zemosaic_filter_gui.py` est retiré dans la première vague, le faire ici, pas avant.

4. **Nettoyage des textes produit**
   - Plus de mentions “Tk stable”, “fallback Tk”, “legacy alternative” dans le frontend officiel.

### Critères de done
- Plus aucune référence active à Tk dans le runtime officiel.
- Anciennes configs chargées sans plantage.
- Sauvegarde de config réécrit dans le nouveau modèle.

### Note importante
S3 ne signifie pas encore forcément “zéro Tk dans tout le repo”. Il signifie “zéro Tk vivant dans l’application officielle”.

---

## S4 — Packaging, distribution, documentation

**Durée estimée:** 2–3 jours  
**Effort estimé:** 2–4 JH  
**Priorité:** P1

### Objectifs
- Rendre la bascule cohérente du point de vue utilisateur et release.

### Tickets
1. **Packaging cleanup**
   - Mettre à jour spec, scripts de build, dépendances, exclusions éventuelles.
   - Vérifier qu’aucun packaging n’embarque Tk par héritage inutile sur le chemin officiel.

2. **Documentation utilisateur**
   - README / wiki / quickstart / troubleshooting.
   - Refaire les sections d’installation GUI autour de PySide6.
   - Supprimer les guides qui expliquent encore la coexistence Tk/Qt sauf archive legacy volontaire.

3. **Release notes et breaking change**
   - Expliquer clairement :
     - que Qt est désormais l’unique frontend officiel ;
     - ce qui est migré automatiquement ;
     - ce qui n’est plus supporté ;
     - quoi faire si PySide6 manque.

### Critères de done
- Build propre.
- Docs synchronisées.
- Release notes prêtes.

---

## S5 — Validation finale release

**Durée estimée:** 2 jours  
**Effort estimé:** 2–3 JH  
**Priorité:** P0 avant release

### Objectifs
- Verrouiller l’absence de régression visible pour l’utilisateur officiel.

### Tickets
1. **Smoke tests multi-OS**
   - Windows
   - Linux
   - macOS si support officiel

2. **Tests de migration config**
   - Charger ancienne config contenant les marqueurs Tk.
   - Vérifier réécriture/migration sans plantage.

3. **Tests de dépendances manquantes**
   - PySide6 absent
   - worker indisponible
   - erreur de lancement GUI
   - vérifier qu’aucun chemin n’essaie de recréer une boîte Tk

4. **Tests headless / worker / config round-trip**
   - valider `import zemosaic_config` dans un env sans Tk ;
   - valider `load_config()` puis `save_config()` ;
   - valider l’idempotence du round-trip ;
   - valider `import zemosaic_worker` sans réintroduction Tk sur les chemins non-GUI.

5. **Workflow QA final**
   - démarrage GUI
   - sélection des dossiers
   - chargement/sauvegarde config
   - pipeline
   - logs
   - fermeture propre

### Critères de done
- QA sign-off.
- Aucun fallback Tk observé.
- Aucun import Tk latent observé via `zemosaic_config` / worker sur les chemins headless validés.
- Release gate validée.

---

## S6 — Purge complète de Tk du repository (optionnel mais propre)

**Durée estimée:** 2–5 jours  
**Effort estimé:** variable selon les outils annexes  
**Priorité:** P2 après release Qt-only stabilisée

### À ne lancer que si la décision est explicite
Cette phase n’est pas une conséquence automatique de S5. Elle nécessite une décision explicite.

### Objectifs
- Supprimer Tk des outils annexes.
- Réduire encore la maintenance globale si cela vaut réellement le coût.

### Candidats typiques
- `lecropper.py`
- `zequalityMT.py`
- legacy filter Tk restant
- scripts ou helpers secondaires encore liés à Tk

### Options possibles
- **Option 1:** suppression pure et simple si l’outil n’est plus utile
- **Option 2:** portage Qt si l’outil reste stratégique
- **Option 3:** gel en outil legacy séparé, non packagé, non supporté officiellement

### Critères de done
- Décision explicite prise pour chaque outil.
- Zéro import Tk dans tout le repo **ou** justification documentée des exceptions restantes.

---

## 6) Risques et mitigations

### Risque 1 — Parité fonctionnelle surestimée
**Problème:** on croit Qt prêt alors qu’un comportement utilisateur dépend encore du legacy.  
**Mitigation:** S1 obligatoire avec matrice détaillée, pas un simple feeling.

### Risque 2 — Suppression trop large, trop tôt
**Problème:** on casse des outils annexes sans bénéfice release immédiat.  
**Mitigation:** scope primaire vs secondaire clairement séparé.

### Risque 3 — Faux Qt-only
**Problème:** Tk n’est plus visible dans l’UI, mais reste utilisé pour les erreurs de lancement.  
**Mitigation:** S2 impose l’absence de Tk en nominal **et** en erreur.

### Risque 4 — Migration config sale
**Problème:** anciennes configs provoquent des comportements ambigus ou des bugs silencieux.  
**Mitigation:** migration testée, rewrite propre, suppression progressive des clés legacy.

### Risque 5 — `zemosaic_config.py` faussement “innocent”
**Problème:** le launcher semble Qt-only mais `zemosaic_config.py` réimporte encore Tk par la porte latérale.  
**Mitigation:** audit S0 dédié + obligation S2 d’un module import-safe sans Tk.

### Risque 6 — Régression fonctionnelle invisible
**Problème:** une feature encore vivante côté backend disparaît parce qu’elle était masquée en Qt, sans décision explicite.  
**Mitigation:** inventaire S0 + décision S1 “fermer / hors-scope / legacy assumé”.

### Risque 7 — Packaging incohérent
**Problème:** le binaire ou l’install continue d’embarquer une logique legacy inutile.  
**Mitigation:** S4 + smoke tests S5.

---

## 7) KPIs de succès

### KPIs minimum pour déclarer victoire sur le scope primaire
- 0 import/runtime Tk sur le chemin officiel de lancement.
- 0 fallback Tk en cas d’erreur de démarrage.
- 0 choix Tk exposé dans l’UI officielle.
- 100% des workflows critiques officiels validés sous Qt.
- `zemosaic_config.py` importable sans Tk sur les chemins officiels et headless validés.
- Toute feature backend encore vivante mais non exposée en Qt est documentée comme fermée ou hors-scope.
- Config legacy migrée sans plantage.

### KPIs bonus
- Simplification packaging.
- Simplification documentation.
- Réduction du périmètre de test GUI.
- Réduction du code mort / des branches conditionnelles liées au backend.

---

## 8) Recommandations pratiques de mise en œuvre

1. **Faire petit, chirurgical, vérifiable**  
   Pas de grand refactor esthétique au milieu de cette migration.

2. **Traiter d’abord le runtime officiel, pas les utilitaires**  
   C’est là que la valeur produit est réelle.

3. **Changer le discours produit en même temps que le code**  
   Si l’UI dit encore “Qt preview”, le produit n’a pas vraiment tourné la page.

4. **Tester l’absence de Tk et pas seulement la présence de Qt**  
   C’est le test qui débusque les faux résidus legacy.

5. **Surveiller spécialement `zemosaic_config.py` et les imports headless**  
   Un faux Qt-only commence souvent par un module de config “banal” qui réimporte Tk.

6. **Ne pas confondre suppression de fichiers et suppression de dépendance**  
   La vraie victoire, c’est le runtime officiel propre.

---

## 9) Conseils spécifiques pour une mission Codex / Junior

Si ce roadmap doit être exécuté par un agent, lui imposer ces règles :

- **No refactor opportuniste.**
- **Diffs chirurgicaux.**
- **Pas de changement de comportement hors scope Tk/Qt.**
- **Toujours mettre à jour `memory.md`.**
- **Toujours cocher les items terminés dans `followup.md` / `agent.md`.**
- **Toujours documenter clairement ce qui a été décidé “in scope” et “out of scope”.**
- **Toujours fournir la preuve de test de lancement sans fallback Tk.**
- **Toujours prouver que `zemosaic_config.py` est importable sans Tk.**
- **Toujours documenter le statut des features backend non exposées en Qt.**

Formulation utile à donner à l’agent :

> Work only on the next unchecked item. Keep changes surgical. Do not refactor unrelated code. Update `memory.md` after each meaningful step. Mark completed items with `[x]` in both `agent.md` and `followup.md`. Explicitly state what remains in scope and out of scope. Prove that the official runtime no longer depends on Tk, including startup error paths.

---

## 10) Verdict

Le cap “Qt-only” est bon.  
La seule manière propre de le réussir est de :
- verrouiller le scope,
- prouver la parité,
- supprimer le fallback runtime,
- migrer la config,
- puis seulement nettoyer le legacy.

Autrement dit : **d’abord rendre Tk inutile, ensuite le retirer**.


---

## 11) Angles morts à couvrir explicitement (ajouts)

Le document est déjà solide. Les points ci-dessous sont les principaux risques résiduels à expliciter pour éviter une “fausse migration terminée”.

### A. Chemin CLI / headless / batch non-GUI
**Risque:** casser involontairement des usages non GUI en retirant du code commun lié à Tk.

**À ajouter au scope:**
- valider explicitement que les commandes non-GUI (si présentes) restent fonctionnelles;
- interdire toute dépendance Qt dans les chemins headless qui n’en ont pas besoin.

**Critère:** “Qt-only frontend” ne doit pas devenir “Qt required for all modes”.

---

### B. Variables d’environnement et flags legacy
**Risque:** des variables historiques (`ZEMOSAIC_GUI_BACKEND`, `--tk-gui`, etc.) continuent de provoquer des comportements ambigus.

**À ajouter au scope:**
- lister toutes les variables/flags backend;
- définir leur statut: supprimé / ignoré avec warning / migré;
- documenter la date de suppression effective.

**Critère:** aucune combinaison de flags ne doit réactiver Tk sur le runtime officiel.

---

### C. Compatibilité scripts externes / automatisation utilisateur
**Risque:** des scripts perso/CI lancent encore le mode Tk (arguments ou env).

**À ajouter au scope:**
- fournir un mapping “ancien flag -> nouveau comportement”;
- émettre un message d’erreur/warning explicite plutôt qu’un échec silencieux.

**Critère:** migration opérable sans reverse-engineering par l’utilisateur.

---

### D. Stratégie de dépréciation et rollback
**Risque:** migration brutale sans filet.

**À ajouter au scope:**
- définir une politique: dépréciation immédiate vs 1 release tampon;
- définir un plan rollback (tag, revert propre, critères de rollback).

**Critère:** décision documentée avant merge final.

---

### E. CI / tests automatiques ciblés “absence Tk”
**Risque:** la QA manuelle oublie un chemin d’import latent.

**À ajouter au scope:**
- job CI qui échoue si import Tk détecté dans le runtime officiel;
- test de démarrage dans un env sans Tk;
- test d’erreur de démarrage sans fallback Tk.

**Critère:** garde-fous automatiques, pas seulement vérification ponctuelle.

---

### F. Frontière “officiel” vs “annexe” trop implicite
**Risque:** zone grise sur `lecropper.py`, `zequalityMT.py`, etc.

**À ajouter au scope:**
- classifier chaque outil annexe en `supported`, `internal`, `legacy`, `deprecated`;
- indiquer packaging oui/non + support oui/non.

**Critère:** chaque module Tk restant est justifié noir sur blanc.

---

### G. Impact packaging réel (pas seulement scripts)
**Risque:** Tk encore embarqué indirectement via hooks/hiddenimports.

**À ajouter au scope:**
- vérifier artefacts build finaux (contenu embarqué) et non juste la config build;
- vérifier taille binaire avant/après et dépendances runtime.

**Critère:** preuves d’artefact sans dépendance Tk côté frontend officiel.

---

### H. Migration configuration: idempotence et nettoyage progressif
**Risque:** migration partielle qui régresse à la sauvegarde suivante.

**À ajouter au scope:**
- migration idempotente (2e lancement sans effet secondaire);
- suppression progressive des clés legacy avec fenêtre de compatibilité;
- tests de round-trip load/save.

**Critère:** ancienne config -> nouvelle config stable en 1 lancement.

---

### I. Documentation opérateur/dev, pas seulement utilisateur final
**Risque:** l’équipe continue d’utiliser des instructions obsolètes (Tk fallback, flags legacy).

**À ajouter au scope:**
- mettre à jour docs dev/build/contribution;
- ajouter une note “Tk retirement” dans guidelines internes.

**Critère:** aucune doc active ne recommande Tk pour le flux officiel.

---

### J. Gouvernance des exceptions temporaires
**Risque:** exceptions “temporaires” qui deviennent permanentes.

**À ajouter au scope:**
- toute exception doit avoir propriétaire + date cible de retrait;
- suivre ces exceptions dans un fichier de suivi release.

**Critère:** zéro exception non datée/non propriétaire.

---

## 12) Gate additionnelle recommandée avant clôture

Ajouter une gate finale binaire:

- **GO** si:
  - runtime officiel démarre et gère ses erreurs sans Tk,
  - `zemosaic_config.py` et les imports headless validés ne dépendent plus de Tk,
  - CI “no-Tk-on-official-path” verte,
  - migration config idempotente validée,
  - docs runtime officielles nettoyées.

- **NO-GO** si un de ces 5 points manque, même si le reste est terminé.



### K. `zemosaic_config.py` import-safe ou faux sentiment de succès
**Risque:** le runtime semble propre, mais un import de config réintroduit encore Tk sur un chemin officiel ou headless.

**À ajouter au scope:**
- exiger explicitement un test `import zemosaic_config` dans un environnement sans Tk;
- décider noir sur blanc entre module Tk-free, split de module ou lazy import strictement hors scope officiel;
- considérer cet item comme un bloqueur P0, pas comme une finition.

**Critère:** aucun import de config officiel/headless ne dépend de Tk.

---

### L. Features backend vivantes mais cachées côté Qt
**Risque:** une fonctionnalité est encore présente dans le worker/backend mais n’est plus visible côté Qt; la suppression de Tk crée alors une régression “silencieuse”.

**À ajouter au scope:**
- inventorier ces features avant retrait Tk;
- exiger une décision explicite pour chacune: exposer, abandonner, classer legacy;
- refuser toute disparition implicite non documentée.

**Critère:** aucune feature backend encore vivante ne disparaît sans décision écrite.

---

## 13) Nouveaux points potentiellement problématiques (après enrichissements)

Les ajouts sont pertinents, mais ils introduisent quelques risques de pilotage qu’il vaut mieux verrouiller pour éviter l’effet “roadmap trop large pour exécution linéaire”.

### M. Risque de double comptage / redondance des exigences
**Constat:** certains points sont désormais exprimés à plusieurs endroits (notamment `zemosaic_config.py` import-safe et features backend non exposées en Qt).

**Risque:** implémentations répétées, discussions infinies en review, difficulté à savoir quand un item est réellement “done”.

**Action recommandée:**
- désigner une **source de vérité unique** par sujet (ex: S2 pour config import-safe, S0/S1 pour backend hidden features) ;
- dans les autres sections, référencer explicitement cette source plutôt que reformuler.

**Critère:** 1 owner + 1 section canonique par exigence critique.

---

### N. Risque de scope creep par accumulation de gates
**Constat:** le plan couvre maintenant produit, runtime, packaging, CI, docs, gouvernance exceptions, rollback, scripts externes, etc.

**Risque:** gel prolongé de la release Qt-only si tout est traité comme bloquant simultanément.

**Action recommandée:**
- figer une liste de **bloquants release P0** (max 5–7 items) ;
- reclasser le reste en P1/P2 planifiés post-release.

**Critère:** GO/NO-GO basé uniquement sur une short-list stable et explicitement versionnée.

---

### O. Risque d’ambiguïté sur le périmètre “headless validé”
**Constat:** le document exige la validité headless, mais ne définit pas précisément quels parcours headless sont “officiels”.

**Risque:** divergences de lecture entre dev/QA (test incomplet vs test trop large).

**Action recommandée:**
- lister les parcours headless officiellement supportés (ex: import config, load/save config, import worker, exécution CLI X/Y) ;
- déclarer explicitement les parcours non supportés.

**Critère:** matrice de tests headless bornée et approuvée.

---

### P. Risque de migration config incomplète multi-plateforme
**Constat:** la migration est bien couverte conceptuellement, mais les chemins utilisateurs et historiques diffèrent selon OS/versions.

**Risque:** migration OK sur une machine de dev mais cassée sur d’anciens environnements.

**Action recommandée:**
- définir un set minimal de fixtures de config legacy (Windows/Linux/macOS, anciennes clés backend/env) ;
- exiger un test de migration sur ces fixtures dans S5.

**Critère:** taux de migration réussi 100% sur fixtures officielles.

---

### Q. Risque “tooling agent” trop couplé à des fichiers locaux non universels
**Constat:** la section “mission Codex / Junior” impose `memory.md`, `followup.md`, `agent.md`.

**Risque:** confusion si ces fichiers n’existent pas dans tous les contextes d’exécution.

**Action recommandée:**
- préciser “si présents” ou fournir un fallback standard (ex: `MIGRATION_LOG.md`) ;
- ne pas bloquer l’exécution sur des conventions locales non garanties.

**Critère:** consignes agent exécutables dans n’importe quel clone standard du repo.

---

### R. Risque de non-alignement versioning/release semantics
**Constat:** breaking change actée, mais le document ne fixe pas la stratégie de version (semver) associée.

**Risque:** communication release ambiguë pour les utilisateurs/outils d’automatisation.

**Action recommandée:**
- fixer explicitement la politique version (ex: minor vs major pour retrait Tk officiel) ;
- lier cette décision à S4 release notes.

**Critère:** numéro de version cible validé avant freeze release.

---

### S. Risque de validation “fonctionnelle” sans mesure de simplification réelle
**Constat:** l’objectif inclut réduction de complexité/maintenance, mais sans métrique de dette supprimée.

**Risque:** migration déclarée réussie sans gain structurel mesurable.

**Action recommandée:**
- ajouter 2 métriques simples avant/après:
  - nombre de branches backend Tk/Qt actives sur runtime officiel,
  - nombre de modules packaging/runtime officiels dépendant de Tk.

**Critère:** baisse objective observée et documentée en clôture.

---

## 14) Recommandation de pilotage (très concrète)

Pour éviter les effets de bord, piloter avec 3 artefacts minimaux:

1. **Checklist P0 Release Qt-only** (courte, bloquante, versionnée)
2. **Register des exceptions temporaires** (owner + échéance)
3. **Matrice de tests officiels** (GUI + headless bornés)

Si ces 3 artefacts sont tenus à jour, le plan reste exigeant sans devenir ingérable.


---

## C. Appendice de conservation — contenu detaille source 2

> Source: ROADMAP_REMOVE_TKINTER_V4.md

# ZeMosaic — Roadmap V4 durci : retrait de Tkinter legacy avec traitement explicite de `lecropper`

Date: 2026-03-13  
Owner: Tristan / ZeMosaic core  
Statut: version V4 — exécutable, avec décision explicite sur le cas `lecropper`

---

## 1) Décision d’architecture à graver noir sur blanc

Le chantier doit être découpé en **trois sujets distincts** :

1. **Objectif A — Qt-only pour l’application officielle ZeMosaic**  
   Cela couvre le lanceur, la fenêtre principale, la fenêtre de filtre, la config, les erreurs de démarrage, le packaging officiel et la documentation officielle.

2. **Objectif B — suppression des dépendances Tk sur tous les chemins officiels et headless validés**  
   Cela inclut non seulement l’UI visible, mais aussi les imports indirects via les modules communs, le worker et la config.

3. **Objectif C — sort des outils annexes encore en Tk**  
   Cela concerne les utilitaires standalone ou semi-standalone comme `lecropper.py`, `zequalityMT.py`, et tout wrapper legacy hors frontend officiel.

Le point critique est le suivant :

> “Qt-only officiel” n’implique pas automatiquement “zéro Tk dans tout le repo”.  
> En revanche, il impose **zéro dépendance Tk sur le runtime officiel et sur les parcours headless validés**.

---

## 2) Décision explicite sur `lecropper`

### Position V4
`lecropper.py` est traité comme un **outil annexe stratégique potentiel**, pas comme un composant du frontend officiel ZeMosaic.

### Conséquence immédiate
Même s’il reste autonome en Tk pendant un temps, il doit cesser d’être une **dépendance du runtime officiel**.  
Autrement dit :

- il peut survivre temporairement comme outil séparé ;
- mais il ne doit plus être importé, directement ou indirectement, par les chemins officiels ou headless validés.

### Politique recommandée
- **Court terme (bloquant P0):** le découpler totalement du worker/runtime officiel.
- **Moyen terme (P1/P2):** décider s’il reste :
  - `supported standalone`,
  - `legacy standalone`,
  - ou `deprecated`.
- **Long terme (option préférée si l’outil reste utile):** portage Qt dédié.

### Recommandation franche
Oui, **c’est un bon moment pour décider sa conversion**, mais **pas** pour en faire un prérequis caché de la bascule Qt-only officielle.  
La bonne séquence est :

1. couper toute dépendance runtime officielle vers `lecropper`,
2. stabiliser la release Qt-only,
3. puis porter `lecropper` vers Qt dans une phase dédiée si l’outil mérite de survivre officiellement.

### Pourquoi cette position est saine
Si tu tentes de faire en même temps :
- la migration Qt-only officielle,
- la purge Tk,
- et le portage complet de `lecropper`,

tu augmentes fortement le risque de scope creep, de review confuse et de régressions sans bénéfice immédiat pour l’utilisateur principal.

---

## 3) Scope retenu

### Scope primaire obligatoire
Le chemin officiel ZeMosaic doit devenir **100% Qt** pour :

- `run_zemosaic.py`
- `zemosaic_gui_qt.py`
- `zemosaic_filter_gui_qt.py`
- `zemosaic_config.py`
- les messages d’erreur de démarrage
- les scripts/builds/docs du frontend officiel
- la migration de configuration utilisateur

### Scope primaire obligatoire élargi
Les parcours headless officiellement validés doivent être **Tk-free** eux aussi, au minimum pour :

- `import zemosaic_config`
- `load_config()` / `save_config()`
- round-trip config idempotent
- `import zemosaic_worker` sur les parcours non-GUI supportés
- toute CLI officielle réellement supportée

### Scope secondaire explicite
Outils annexes ou legacy :

- `zemosaic_gui.py`
- `zemosaic_filter_gui.py`
- `lecropper.py`
- `zequalityMT.py`
- autres wrappers/helpers Tk hors chemin officiel

### Hors-scope par défaut de la première vague
- portage Qt complet de `lecropper`
- suppression totale de Tk du repository
- refactor esthétique large
- réorganisation opportuniste de modules non nécessaires à la bascule officielle

---

## 4) Nouvelle règle de gouvernance

Chaque module Tk encore présent à la fin de la release Qt-only doit avoir une fiche simple :

- **statut**: `supported` / `internal` / `legacy` / `deprecated`
- **packagé**: oui / non
- **support officiel**: oui / non
- **owner**
- **date cible de retrait ou de conversion**
- **justification**

Sans cette fiche, l’exception n’est pas valide.

---

## 5) Principe directeur

La bonne définition de “Qt-only officiel” est la suivante :

> Un utilisateur qui lance ZeMosaic par le chemin officiel ne doit dépendre de Tk ni en cas nominal, ni en cas d’erreur de démarrage, ni dans sa config, ni dans l’UI, ni dans les imports headless officiellement validés.

Corollaire V4 :

> Un outil annexe Tk est acceptable temporairement **uniquement** s’il est réellement isolé du runtime officiel.

Donc `lecropper` ne pose plus problème **s’il devient vraiment standalone**.  
Il redevient problématique dès qu’il fuit dans le worker, la config, le packaging officiel ou les tests headless officiels.

---

## 6) Source de vérité unique par exigence critique

Pour éviter les redondances et les reviews interminables :

- **Parité fonctionnelle Qt** → source canonique = **S1**
- **Runtime officiel sans Tk** → source canonique = **S2**
- **Migration config** → source canonique = **S3**
- **Packaging/docs/release** → source canonique = **S4**
- **Validation multi-OS et headless** → source canonique = **S5**
- **Sort de `lecropper` et des outils annexes** → source canonique = **S6**

Les autres sections peuvent rappeler les enjeux, mais ne doivent pas redéfinir les critères de done.

---

## 7) Ordre impératif

**S0 → S1 → S2 → S3 → S4 → S5**  
Puis éventuellement **S6** pour les outils annexes et la purge complète.

Interdit de lancer un portage Qt de `lecropper` au milieu de S1/S2 si cela retarde ou brouille la bascule officielle.

---

## S0 — Audit verrouillé et bornage du scope

**Durée estimée:** 2–3 jours  
**Effort estimé:** 2–4 JH  
**Priorité:** P0

### Objectifs
- Geler le périmètre exact.
- Inventorier tous les usages Tk.
- Distinguer clairement officiel / headless validé / annexe / legacy.
- Trancher le statut initial de `lecropper`.

### Tickets
1. **Inventaire exhaustif des imports et usages Tk**
   - rechercher `tkinter`, `messagebox`, `filedialog`, `ttk`, `Tk()`, `FigureCanvasTkAgg`, wrappers et imports transitifs ;
   - produire une table par fichier avec catégorie :
     - `official runtime path`
     - `official headless path`
     - `legacy GUI`
     - `standalone utility`
     - `build/doc/test only`

2. **Audit du chemin officiel**
   - vérifier lancement GUI nominal ;
   - vérifier démarrage avec PySide6 absent ;
   - vérifier erreur d’import worker ;
   - vérifier erreur de lancement GUI ;
   - lister tout chemin qui réintroduit Tk.

3. **Audit critique config**
   - identifier tous les imports Tk directs et indirects dans `zemosaic_config.py` ;
   - décider noir sur blanc entre :
     - module Tk-free,
     - split `config_core` / `config_legacy_gui`,
     - lazy import strictement hors chemin officiel.

4. **Audit headless borné**
   - lister les parcours headless officiellement supportés ;
   - déclarer explicitement les parcours non supportés ;
   - interdire toute extension implicite du scope headless pendant l’exécution.

5. **Décision initiale sur `lecropper`**
   - classer `lecropper` dans l’une des catégories :
     - `supported standalone candidate for Qt port`
     - `legacy standalone`
     - `deprecated`
   - exiger une décision distincte sur :
     - support officiel oui/non,
     - packaging officiel oui/non,
     - portage Qt oui/non plus tard.

6. **Matrice de parité Qt**
   - lister les workflows critiques ;
   - marquer `OK`, `à compléter`, `bloquant`, `hors-scope`.

### Critères de done
- scope primaire validé par écrit ;
- headless validé borné par écrit ;
- stratégie config décidée ;
- statut initial de `lecropper` décidé ;
- liste P0/P1 stabilisée.

### Interdits
- pas de suppression de code Tk en S0 ;
- pas de portage Qt de `lecropper` en S0 ;
- pas de refactor opportuniste.

---

## S1 — Parité fonctionnelle Qt stricte

**Durée estimée:** 4–6 jours  
**Effort estimé:** 4–8 JH  
**Priorité:** P0

### Objectifs
- Garantir qu’aucun workflow critique officiel n’exige encore Tk.
- Éliminer le discours “Qt preview”.

### Tickets
1. **Fermer les gaps fonctionnels Qt**
   - réglages principaux ;
   - dialogues critiques ;
   - logs / feedback ;
   - callbacks de démarrage / arrêt ;
   - comportements liés au filtre si celui-ci fait partie du frontend officiel.

2. **Parité de persistance config**
   - vérifier les réglages réellement sauvegardés/relus ;
   - supprimer les ambiguïtés entre UI Qt et config persistée.

3. **Nettoyage UX préparatoire**
   - retirer “preview” ;
   - préparer la disparition du choix de backend.

4. **Features backend vivantes mais cachées**
   - pour chaque fonctionnalité non exposée côté Qt :
     - exposer,
     - assumer hors-scope,
     - ou classer legacy ;
   - aucune disparition implicite non documentée.

5. **Validation workflows critiques**
   - démarrage GUI ;
   - chargement / sauvegarde config ;
   - pipeline standard ;
   - Grid mode ;
   - SDS si concerné ;
   - outils officiellement exposés ;
   - fermeture propre ;
   - logs / export.

### Critères de done
- matrice de parité complétée ;
- aucun blocant P0/P1 restant pour le frontend officiel ;
- Qt n’est plus présenté comme expérimental.

### Interdits
- ne pas encore supprimer le fallback runtime tant que S1 n’est pas validé.

---

## S2 — Qt-only au runtime officiel

**Durée estimée:** 2–3 jours  
**Effort estimé:** 2–4 JH  
**Priorité:** P0

### Objectifs
- Faire de Qt le seul frontend officiel.
- Supprimer toute dépendance Tk du chemin officiel nominal et d’erreur.
- Couper toute fuite de `lecropper` ou autre outil Tk vers le runtime officiel.

### Tickets
1. **Supprimer le fallback backend dans `run_zemosaic.py`**
   - plus de bascule implicite Tk ;
   - plus de `--tk-gui` sur le chemin officiel ;
   - plus de backend switch exposé pour l’app officielle.

2. **Supprimer les message boxes Tk de démarrage**
   - erreurs gérées en console, Qt ou mécanisme neutre ;
   - aucune création de root Tk.

3. **Rendre la config import-safe sans Tk**
   - `zemosaic_config.py` importable sans Tk ;
   - si besoin, split clair core/legacy ;
   - aucun import innocent ne doit réintroduire Tk.

4. **Basculer la config officielle sur Qt**
   - valeur par défaut = `qt` ;
   - lecture des anciennes configs assurée ;
   - préparer la neutralisation de `preferred_gui_backend`.

5. **Supprimer le choix de backend dans l’UI Qt**
   - plus de groupe backend ;
   - plus de copie “Tk stable / Qt preview”.

6. **Durcir les messages de dépendances**
   - PySide6 est requis pour le frontend officiel ;
   - ne plus suggérer d’utiliser l’interface Tk dans les modules officiels.

7. **Découplage P0 de `lecropper`**
   - supprimer tout import direct ou indirect de `lecropper` depuis le runtime officiel ;
   - si un besoin backend réel subsiste, extraire la logique non-UI nécessaire dans un module pur sans Tk ;
   - laisser `lecropper.py` comme coquille standalone temporaire si besoin.

### Critères de done
- application officielle Qt-only ;
- zéro import Tk sur le chemin officiel nominal ;
- zéro import Tk sur le chemin d’erreur de démarrage ;
- zéro import de `lecropper` ou d’autre utilitaire Tk dans le runtime officiel ;
- `zemosaic_config.py` importable sans Tk ;
- utilisateur ne peut plus choisir Tk dans l’UI officielle.

### Gate de sortie obligatoire
Tester explicitement un environnement **sans Tk disponible** et vérifier :
- que le lancement officiel fonctionne ou échoue proprement sans fallback Tk ;
- que `import zemosaic_config` ne réintroduit pas Tk ;
- que `import zemosaic_worker` sur les parcours headless validés ne réintroduit pas Tk ;
- que l’absence de `lecropper` ne casse pas le runtime officiel.

---

## S3 — Migration config et nettoyage officiel

**Durée estimée:** 3–5 jours  
**Effort estimé:** 4–7 JH  
**Priorité:** P0

### Objectifs
- Nettoyer la dette Tk morte dans le périmètre officiel.
- Garder une migration utilisateur propre et idempotente.

### Tickets
1. **Supprimer les branches mortes liées au backend**
   - flags ;
   - conditions ;
   - docstrings ;
   - commentaires legacy.

2. **Migration de configuration legacy**
   - rewriter `preferred_gui_backend = tk` vers `qt` ;
   - neutraliser puis retirer les clés devenues sans sens ;
   - garantir l’idempotence ;
   - tester le round-trip load/save.

3. **Nettoyage des modules legacy du frontend officiel**
   - retirer du packaging/runtime officiel ce qui n’est plus callable ;
   - ne pas confondre cela avec la purge complète du repo.

4. **Nettoyage des textes produit**
   - plus de mentions Tk dans le flux officiel.

### Critères de done
- anciennes configs chargées sans plantage ;
- migration stable en un lancement ;
- sauvegarde conforme au nouveau modèle ;
- plus aucune référence active à Tk dans le runtime officiel.

---

## S4 — Packaging, distribution, documentation

**Durée estimée:** 2–3 jours  
**Effort estimé:** 2–4 JH  
**Priorité:** P1

### Objectifs
- Rendre la bascule cohérente côté utilisateur, build et release.

### Tickets
1. **Packaging cleanup**
   - mettre à jour spec et scripts ;
   - vérifier les artefacts build finaux ;
   - vérifier taille et dépendances runtime avant/après ;
   - prouver que Tk n’est plus embarqué par héritage inutile sur le frontend officiel.

2. **Documentation utilisateur**
   - README / wiki / quickstart / troubleshooting ;
   - installation autour de PySide6 ;
   - plus de coexistence Tk/Qt dans la doc officielle.

3. **Documentation opérateur/dev**
   - contribution/build/dev notes ;
   - note interne “Tk retirement” ;
   - mapping des anciens flags/env si nécessaire.

4. **Release notes**
   - Qt est l’unique frontend officiel ;
   - migration automatique ;
   - ce qui n’est plus supporté ;
   - statut des outils annexes, notamment `lecropper`.

### Critères de done
- build propre ;
- docs synchronisées ;
- release notes prêtes ;
- statut public de `lecropper` clair.

---

## S5 — Validation finale release

**Durée estimée:** 2 jours  
**Effort estimé:** 2–3 JH  
**Priorité:** P0 avant release

### Objectifs
- Verrouiller l’absence de régression visible pour l’utilisateur officiel.

### Tickets
1. **Smoke tests multi-OS**
   - Windows ;
   - Linux ;
   - macOS si support officiel.

2. **Tests de migration config**
   - fixtures legacy multi-plateformes ;
   - anciennes clés backend/env ;
   - vérification de la réécriture.

3. **Tests de dépendances manquantes**
   - PySide6 absent ;
   - worker indisponible ;
   - erreur de lancement GUI ;
   - vérifier qu’aucun chemin n’ouvre Tk.

4. **Tests headless bornés**
   - `import zemosaic_config` ;
   - `load_config()` / `save_config()` ;
   - round-trip idempotent ;
   - `import zemosaic_worker` sur les parcours supportés ;
   - vérifier qu’aucune dépendance à `lecropper` ne revient.

5. **Workflow QA final**
   - démarrage GUI ;
   - sélection des dossiers ;
   - config ;
   - pipeline ;
   - logs ;
   - fermeture propre.

6. **CI “no-Tk-on-official-path”**
   - job qui échoue si import Tk détecté sur le runtime officiel ;
   - job qui échoue si un parcours headless validé réimporte Tk.

### Critères de done
- QA sign-off ;
- aucun fallback Tk observé ;
- aucun import Tk latent observé sur les chemins officiels et headless validés ;
- release gate validée.

---

## S6 — Outils annexes et sort final de `lecropper`

**Durée estimée:** 2–6 jours  
**Effort estimé:** variable  
**Priorité:** P2 après stabilisation Qt-only

### Précondition
Ne lancer S6 qu’après validation complète de S5.

### Objectifs
- statuer définitivement sur les outils annexes ;
- réduire la maintenance globale ;
- décider si `lecropper` doit survivre et sous quelle forme.

### Options pour `lecropper`
#### Option A — Legacy standalone gelé
- reste en Tk ;
- non packagé avec le frontend officiel ;
- non supporté officiellement sauf mention contraire ;
- clairement documenté comme outil séparé.

#### Option B — Portage Qt dédié
- option **préférée** si `lecropper` reste utile et utilisé ;
- portage dans une mission séparée ;
- extraction préalable de toute logique métier réutilisable dans un module pur ;
- nouvelle UI Qt propre ;
- tests indépendants ;
- packaging décidé explicitement.

#### Option C — Dépréciation puis retrait
- si l’outil n’apporte plus assez de valeur ;
- communication claire ;
- fenêtre de transition éventuelle.

### Recommandation V4
Par défaut, choisir :

- **S2:** découplage obligatoire ;
- **S6:** **Option B** si `lecropper` est jugé encore stratégique ;
- sinon **Option A** transitoire puis décision finale.

### Critères de done
- décision explicite prise pour `lecropper` ;
- owner et échéance définis ;
- si portage Qt : parité minimale et tests dédiés ;
- si legacy : isolement réel et non-packaging officiel ;
- si retrait : doc et release notes alignées.

---

## 8) Risques et mitigations

### Risque 1 — Faux standalone
**Problème:** `lecropper` est présenté comme autonome mais continue de fuiter dans le worker ou le runtime officiel.  
**Mitigation:** découplage P0 en S2.

### Risque 2 — Scope creep
**Problème:** la conversion de `lecropper` se mélange à la migration Qt-only officielle.  
**Mitigation:** portage éventuel repoussé en S6.

### Risque 3 — Faux Qt-only
**Problème:** l’UI officielle semble Qt-only mais les erreurs de démarrage, la config ou les imports headless réintroduisent Tk.  
**Mitigation:** S2 + S5 + CI dédiée.

### Risque 4 — Migration config sale
**Problème:** anciennes configs ambiguës ou non idempotentes.  
**Mitigation:** S3 + fixtures S5.

### Risque 5 — Exceptions éternelles
**Problème:** modules Tk restants sans owner ni échéance.  
**Mitigation:** fiche d’exception obligatoire.

### Risque 6 — Packaging incohérent
**Problème:** Tk reste embarqué par inertie dans les builds officiels.  
**Mitigation:** vérification des artefacts finaux en S4.

---

## 9) P0 release checklist courte et bloquante

La release Qt-only officielle est **GO** seulement si :

1. le runtime officiel démarre et gère ses erreurs sans Tk ;
2. `zemosaic_config.py` est importable sans Tk ;
3. les parcours headless validés n’importent pas Tk ;
4. `lecropper` et autres outils Tk ne sont plus des dépendances du runtime officiel ;
5. la migration config legacy est idempotente ;
6. les docs et release notes officielles sont alignées ;
7. la CI “no-Tk-on-official-path” est verte.

Sinon : **NO-GO**.

---

## 10) Conseils d’exécution pour Codex / Junior

- Work only on the next unchecked item.
- Keep changes surgical.
- Do not refactor unrelated code.
- Update `memory.md` after each meaningful step.
- Mark completed items with `[x]` in both `agent.md` and `followup.md`.
- If these files do not exist, use `MIGRATION_LOG.md`.
- Explicitly state what remains in scope and out of scope.
- Prove that the official runtime no longer depends on Tk.
- Prove that `lecropper` is either isolated, ported, or explicitly classified legacy/deprecated.

---

## 11) Verdict V4

Le cap reste le bon :

- **d’abord** rendre Tk inutile sur tout le runtime officiel ;
- **ensuite** nettoyer le legacy officiel ;
- **puis seulement** décider du sort final des outils annexes ;
- et, pour `lecropper`, **le bon réflexe maintenant est de le découpler tout de suite et de porter en Qt plus tard si tu veux le conserver officiellement**.

Autrement dit :

> **oui pour préparer la conversion de `lecropper`, non pour mélanger cette conversion avec le cœur du cutover Qt-only officiel.**