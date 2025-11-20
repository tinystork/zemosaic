# Follow-up — Vérifications après patch de synchronisation SDS (Qt main ↔ Qt filter)

Merci pour les modifications. Voici la checklist à suivre pour valider que tout est correct.

---

## 1. Revue de diff

Vérifier dans le diff :

1. **zemosaic_filter_gui_qt.py**
   - [x] `DEFAULT_FILTER_CONFIG.setdefault("sds_mode_default", True)` est bien devenu :

     ```python
     DEFAULT_FILTER_CONFIG.setdefault("sds_mode_default", False)
     ```

   - [x] Le calcul de `self._sds_mode_initial` dans `FilterQtDialog.__init__` :
     - prend d’abord `initial_overrides["sds_mode"]` si présent ;
     - sinon, lit `config_overrides["sds_mode_default"]` ;
     - sinon, utilise `DEFAULT_FILTER_CONFIG["sds_mode_default"]` (False).
   - [x] Aucun autre changement fonctionnel non demandé sur le SDS.

2. **zemosaic_gui_qt.py**
   - [x] Dans `_launch_filter_dialog()` :
     - `initial_overrides` contient désormais une clé `"sds_mode"` initialisée depuis `self.config["sds_mode_default"]`.
   - [x] Dans `_apply_filter_overrides_to_config()` :
     - un bloc gère `if "sds_mode" in overrides: self._update_widget_from_config("sds_mode_default", overrides["sds_mode"])`
     - [x] La logique existante autour (`cluster_panel_threshold`, `astap_max_instances`, etc.) est intacte.

---

## 2. Tests fonctionnels

### 2.1. Test de base sans config existante

1. Supprimer ou renommer temporairement le fichier de config utilisateur (`zemosaic_config.json`) pour simuler un premier lancement.
2. Lancer la GUI Qt principale.
3. Vérifier :
   - La case « **Enable SDS mode by default** » est **décochée**.
4. Ouvrir le filter Qt depuis le bouton dédié.
5. Vérifier :
   - La case « **Enable ZeSupaDupStack (SDS)** » est **décochée**.
   - Fermer sans changer cette case pour ce test.

Résultat attendu : **aucune des cases n’est cochée** par défaut.

---

### 2.2. Propagation Main → Filter

1. Dans la GUI principale Qt :
   - Cocher la case « Enable SDS mode by default ».
   - S’assurer que la config est mise à jour (optionnel : sauvegarde ou autre action déjà existante).
2. Ouvrir le filter Qt.
3. Vérifier :
   - La case « Enable ZeSupaDupStack (SDS) » est **cochée** dès l’ouverture.

4. Fermer le filter sans changer la case SDS.

Résultat attendu : **le filter reflète correctement le choix du main**.

---

### 2.3. Propagation Filter → Main

1. Toujours dans la même session, rouvrir le filter Qt.
   - Vérifier que la case SDS est toujours dans l’état attendu.
2. Cette fois, **changer l’état de la case SDS dans le filter** (par exemple, la décocher si elle était cochée).
3. Valider / OK pour fermer le filter (chemin où les overrides sont renvoyés).
4. Revenir à la GUI principale Qt.

Vérifier :

- La case « **Enable SDS mode by default** » dans le *Main* est synchronisée :
  - Si la case a été décochée dans le filter → elle doit être décochée dans le main.
  - Si la case a été cochée dans le filter → elle doit être cochée dans le main.

Résultat attendu : **le Main se cale sur le choix du Filter** via `_apply_filter_overrides_to_config`.

---

### 2.4. Persistance de la préférence SDS

1. Après avoir modifié l’état SDS via le Filter, déclencher une action qui sauvegarde la configuration (par ex. lancement de traitement, ou toute séquence qui déclenche `save_config` dans `zemosaic_gui_qt.py`).
2. Quitter la GUI Qt principale.
3. Relancer ZeMosaic Qt.
4. Vérifier :
   - La case « Enable SDS mode by default » est dans le même état qu’au moment de la dernière fermeture.
5. Ouvrir le filter Qt :
   - Vérifier que la case « Enable ZeSupaDupStack (SDS) » est également dans le même état.

Résultat attendu : **la préférence SDS est persistante** et cohérente entre les deux interfaces.

---

> ⚠️ Tests manuels 2.1–2.4 non exécutés dans cet environnement : l’interface Qt nécessite un contexte graphique interactif.


## 3. Non-régressions

- Vérifier qu’aucun autre comportement SDS n’a été cassé :
  - Si SDS est activé, le pipeline SDS se comporte comme avant (méga-tiles, global plan, etc.).
  - Si SDS est désactivé, le pipeline classique continue de fonctionner normalement.
- Vérifier qu’aucun autre paramètre de filter (clustering, coverage-first, etc.) n’a vu son comportement changer.

Si tous ces tests passent, la synchronisation SDS Qt main ↔ Qt filter est considérée comme **validée**.
````
