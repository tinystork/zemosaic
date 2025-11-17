# FOLLOWUP.md — ZeMosaic Qt filter GUI

Ce document guide Codex pour les itérations successives sur l’interface **PySide6** de ZeMosaic, en s’assurant de rester strictement compatible avec :

- La logique métier existante (zemosaic_worker, zemosaic_utils, etc.)
- Le comportement historique du GUI Tkinter (`zemosaic_filter_gui.py`, `zemosaic_gui.py`)
- Les contraintes de performance (pas de freeze, respect du multi-process, etc.)

Les fichiers principalement concernés sont :

- `zemosaic_filter_gui_qt.py`
- `zemosaic_gui_qt.py`
- (en lecture seule / référence) `zemosaic_filter_gui.py`, `zemosaic_gui.py`, `zemosaic_worker.py`, `zemosaic_utils.py`


---

## 0. Contraintes globales (à respecter pour TOUTE modif)

1. **Ne pas casser Tkinter**  
   - Ne rien modifier dans `zemosaic_gui.py` ni `zemosaic_filter_gui.py` sauf si la tâche le demande explicitement.  
   - Le workflow historique Tk doit continuer à fonctionner exactement comme aujourd’hui.

2. **Ne pas casser le worker / pipeline**  
   - Ne pas changer la signature ni la logique de base de `run_hierarchical_mosaic` / `run_hierarchical_mosaic_process` dans `zemosaic_worker.py`.  
   - Lorsque le filtre Qt appelle le worker, il doit continuer à fournir les mêmes structures de données que le filtre Tk (simplement avec un sous-ensemble de fichiers si nécessaire).

3. **Pas de régression sur les modes d’empilement**  
   - **Ne pas modifier** la sémantique existante de `batch_size` dans le pipeline de stacking (mode `batch_size = 0` et `batch_size > 1` DOIVENT rester inchangés).

4. **Performance & UI**  
   - Toute opération lourde (scan, clustering, pré-plan WCS, etc.) doit rester hors du thread UI (thread / process séparé).  
   - Garder le Sky preview fluide et éviter toute opération O(N_tiles) à chaque event de souris.

5. **Localisation**  
   - Lors de l’ajout de nouveaux textes visibles (labels, tooltips, menus contextuels), utiliser le localizer Qt (`ZeMosaicLocalization`) quand disponible.  
   - Prévoir des clés de localisation explicites (ex. `qt_filter_clear_bbox`) mais **ne pas** modifier les fichiers `en.json` / `fr.json` ici (ce sera fait séparément).


---

## 1. Rappel du comportement actuel souhaité (déjà implémenté)

Cette section résume ce que le code actuel est supposé faire, pour que tu ne le casses pas :

1. **Affichage des groupes via WCS**  
   - Lorsque des fichiers avec WCS sont chargés, la carte du ciel (Sky preview) affiche des points (centres des brutes) colorés par groupe.  
   - Pour limiter les freezes, **seuls les cadres des groupes** (bounding boxes par groupe) sont tracés, pas chaque footprint individuel.

2. **Taille des cadres de groupes**  
   - Chaque cadre de groupe utilise la taille du **premier WCS** de ce groupe comme gabarit (pour refléter la taille réelle d’une master tile sur le ciel).  
   - Le centre du cadre correspond au centre moyen des RA/DEC du groupe (ou cohérent avec la logique déjà implémentée).

3. **Organisation auto des master tiles**  
   - Le bouton **“Auto-organize Master Tiles”** applique la même logique que la version Tk :
     - Clusterisation des brutes.
     - Éventuelle séparation des groupes par orientation.
     - Préparation des groupes & sizes transmis au worker.
   - Sans bounding box utilisateur, toutes les brutes cochées (ou non exclues) sont candidates à l’auto-organisation.

4. **Scan / grouping log**  
   - Le bloc “Scan / grouping log” inutile a été retiré du UI Qt pour alléger l’interface (ne pas le réintroduire).


---

## 2. NOUVELLE SECTION — Gestion de la bounding box utilisateur

Objectif : rendre la **bounding box “Sky Preview” vraiment utile** et intuitive :

- Pouvoir **l’effacer** facilement si elle est mal placée.
- Lorsque la bounding box est active, **ne sélectionner que les images dont le centre tombe à l’intérieur** pour l’auto-organisation.
- S’assurer que le **chemin vers le worker** reste identique à celui où l’on passe une liste complète, simplement filtrée.


### 2.1. Ajouter un mécanisme pour effacer la bounding box

Fichier principal : `zemosaic_filter_gui_qt.py`

1. **État interne de la bounding box**  
   - Identifier où la bounding box de sélection utilisateur est stockée (typiquement quelque chose comme :  
     - coordonnées pixel dans le plot, et/ou  
     - bornes RA/DEC `bbox_ra_min`, `bbox_ra_max`, `bbox_dec_min`, `bbox_dec_max`.  
   - Si ce n’est pas déjà le cas, stocker proprement cet état dans un attribut du dialog (ex : `self._user_bbox_sky = None` ou dict avec les bornes RA/DEC).

2. **Menu contextuel sur le Sky Preview (clic droit)**  
   - Sur la zone du graphique (canvas Matplotlib intégré via `FigureCanvasQTAgg`), ajouter la gestion d’un **clic droit** qui ouvre un menu contextuel Qt.  
   - Menu minimal avec au moins une entrée :
     - Label anglais par défaut : `"Clear selection bounding box"`  
       - passer par le localizer quand disponible, clé suggérée : `qt_filter_clear_bbox`.
   - Quand l’utilisateur clique sur cette entrée :
     - Supprimer la bounding box graphique (rectangle et/ou overlay) de la figure Matplotlib.
     - Réinitialiser l’état interne (ex: `self._user_bbox_sky = None`).
     - Forcer un refresh/redraw du Sky preview pour s’assurer que le cadre disparaît.

3. **Comportement sans bounding box**  
   - Si aucune bounding box utilisateur n’est active (**état `None`**), le comportement doit être strictement identique à la situation actuelle : toutes les brutes coché·es restent candidates pour les opérations (analyse, auto-organize, etc.).
   - Ne jamais filtrer quoi que ce soit quand la bbox est absente.


### 2.2. Filtrer les images par centre RA/DEC lors de “Auto-organize Master Tiles”

Problème actuel :  
Même lorsqu’une bounding box est dessinée, le bouton **“Auto-organize Master Tiles”** sélectionne/organise **toutes** les images.  
Ce que l’on veut : **seules les images dont le centre se trouve dans la bounding box** doivent être considérées comme candidates.

1. **Localiser le point d’entrée de l’auto-organisation Qt**  
   - Dans `zemosaic_filter_gui_qt.py`, trouver la méthode qui gère le clic sur le bouton **“Auto-organize Master Tiles”** (souvent une slot connectée à un `QPushButton`).  
   - Identifier là où est construite la liste des `_NormalizedItem` inclus pour clusterisation / organisation (typiquement une liste filtrée à partir de `self._items` ou similaire).

2. **Détermination de l’inclusion dans la bounding box**  
   - Chaque `_NormalizedItem` possède déjà (ou doit posséder) :  
     - `center_ra_deg: float | None`  
     - `center_dec_deg: float | None`  
   - La bounding box utilisateur doit être exprimée en RA/DEC :  
     - `bbox_ra_min`, `bbox_ra_max` (en degrés)  
     - `bbox_dec_min`, `bbox_dec_max` (en degrés)
   - Ajouter une fonction utilitaire dans le dialog, par exemple :

     ```python
     def _item_inside_user_bbox(self, item: _NormalizedItem) -> bool:
         if self._user_bbox_sky is None:
             return True  # pas de bbox => inclusion totale
         if item.center_ra_deg is None or item.center_dec_deg is None:
             return False
         ra = float(item.center_ra_deg)
         dec = float(item.center_dec_deg)
         ra_min, ra_max = self._user_bbox_sky["ra_min"], self._user_bbox_sky["ra_max"]
         dec_min, dec_max = self._user_bbox_sky["dec_min"], self._user_bbox_sky["dec_max"]
         # gérer RA qui wrappe à 360 si nécessaire
         if ra_min <= ra_max:
             ra_ok = (ra_min <= ra <= ra_max)
         else:
             # bbox qui traverse 0h
             ra_ok = (ra >= ra_min) or (ra <= ra_max)
         dec_ok = (dec_min <= dec <= dec_max)
         return ra_ok and dec_ok
     ```

   - Cette fonction doit être **robuste** et renvoyer `True` pour tous les items lorsque `self._user_bbox_sky` est `None` (pour garder le comportement actuel si aucune bbox n’est en place).

3. **Filtrage avant auto-organisation**  
   - Juste avant de lancer la logique d’auto-organisation (construction des groupes pour ZeSupaDupStack / SDS, clustering, etc.), filtrer la liste des items candidats :

     ```python
     candidates = [it for it in all_items if self._item_inside_user_bbox(it)]
     ```

   - Si, après filtrage :
     - `len(candidates) == 0` :
       - Ne pas lancer d’auto-organisation.
       - Afficher un message d’avertissement (log + éventuellement QMessageBox) du type :  
         - `"No frames found inside the current selection bounding box."` (clé possible : `qt_filter_bbox_empty_selection`).
   - Les **groupes** passés au worker doivent être formés **uniquement** à partir de ces `candidates`.

4. **Cohérence avec la vue en arbre (treeview)**  
   - Lorsque l’auto-organisation est déclenchée avec une bbox active, les brutes en dehors de la bbox ne doivent pas être sélectionnées dans la vue en arbre (coches/selection).  
   - Adapter la mise à jour du `QTreeWidget` :
     - Cocher / marquer les items qui appartiennent à un groupe retenu et sont à l’intérieur de la bbox.  
     - Laisser décochés ceux qui sont hors bbox (même s’ils appartiennent à un cluster théorique global).

5. **Compatibilité avec les groupes existants**  
   - Si la logique de regroupement s’appuie déjà sur des groupes pré-calculés (cluster connectés) :
     - Filtrer les groupes en ne gardant que ceux qui contiennent **au moins un item dans la bbox**.
     - À l’intérieur de chaque groupe retenu, ne conserver que les items dans la bbox pour le passage au worker.


### 2.3. Respect du “chemin worker” avec bounding box

Problème exprimé :  
> “vérifier que dans le cadre de l'utilisation de la boundbox le chemin du worker est honoré comme dans le cas ou une liste complète est passée par ce biais”

Concrètement : lorsque le filtre Qt utilise la bounding box pour réduire le jeu de données, le **chemin d’appel vers le worker** doit rester **strictement le même**, uniquement avec un **sous-ensemble** de fichiers :

1. **Ne pas changer la signature de l’appel au worker**  
   - La fonction (ou méthode) qui prépare l’appel à `run_hierarchical_mosaic_process` (via le main Qt) ne doit pas changer de signature.
   - Elle doit recevoir la même structure qu’avant (`selected_entries`, `groups`, `config`, etc.), uniquement avec **moins d’entrées**.

2. **Filtrage en amont uniquement**  
   - Toute logique liée à la bounding box doit rester **dans le filtre Qt** (`zemosaic_filter_gui_qt.py`).  
   - Le worker (`zemosaic_worker.py`) ne doit pas être modifié pour cette fonctionnalité : il doit simplement recevoir une liste plus courte et se comporter comme d’habitude.

3. **Validation comportementale**  
   - Après implémentation :
     - Cas A : pas de bounding box → l’auto-organisation doit produire exactement le même pré-plan qu’avant (même nombre de groupes, même log “Prepared N group(s), sizes: [...]”).  
     - Cas B : bounding box couvrant seulement une partie du champ →  
       - Le log doit montrer moins de groupes ou des tailles plus petites.  
       - Le worker doit s’exécuter sans erreur, avec les mêmes chemins de sortie qu’avant.  
       - Aucune différence dans le type/structure des données échangées, hormis le nombre de brutes.

4. **Log de debug optionnel**  
   - Pour faciliter les tests, ajouter un log INFO côté filtre Qt lors de l’appel à l’auto-organisation quand une bbox est active, par exemple :  

     ```text
     [QtFilter] Bounding box active: kept X / Y frames for auto-organize.
     ```

   - Ne pas spammer le log (un message par clic sur le bouton suffit).


---

## 3. Tests manuels à effectuer

Après les changements, valider au minimum les scénarios suivants :

1. **Sans bounding box**  
   - Charger un jeu de brutes avec WCS.  
   - Ne pas dessiner de bounding box.  
   - Cliquer sur **Auto-organize Master Tiles**.  
   - Vérifier :
     - Que le comportement est identique à la version précédente (nombre de groupes, sélection dans l’arbre, log, absence de freeze).

2. **Bounding box partielle**  
   - Dessiner une bounding box englobant seulement une partie des points du Sky preview.  
   - Cliquer sur **Auto-organize Master Tiles**.  
   - Vérifier :
     - Seules les brutes dont le **centre** tombe dans la bbox sont sélectionnées et passées au worker.  
     - Le log montre un nombre de groupes et de frames **réduit** par rapport au cas sans bbox.  
     - La mosaïque produite couvre uniquement la zone de la bbox (dans la logique ZeMosaic).

3. **Effacement de la bounding box**  
   - Dessiner une bbox.  
   - Clic droit dans le Sky preview → sélectionner “Clear selection bounding box”.  
   - Vérifier :
     - Le cadre disparaît visuellement.  
     - L’état interne est bien réinitialisé (prochaine auto-organisation = comportement “global”).

4. **Bord de champ / RA wrap**  
   - Si possible, tester un dataset où la bbox traverse 0h ou frôle les bords du champ.  
   - Vérifier que la gestion RA min/max ne exclut pas à tort des frames.


---

Fin du fichier `followup.md`.
