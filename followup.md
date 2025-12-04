# 🔁 Follow-up : Validation & Ajustements

Ceci est la liste des vérifications à effectuer sur votre implémentation.  
Cochez les cases lors des itérations :

## 🔧 Implémentation générale
- [x] Le pipeline classique est intact (aucune différence dans les logs/classiques)
- [x] `detect_grid_mode()` bascule proprement sans effet secondaire
- [x] `run_grid_mode()` est complètement isolé

## 📥 Lecture du stack_plan.csv
- [x] Fonction de parsing robuste
- [x] Colonnes ignorées correctement
- [x] Paths vérifiés

## 🌐 Construction de la grille
- [x] WCS global stable
- [x] Conversion RA/Dec → X,Y correcte
- [x] Grille régulière générée avec overlap

## 🎛 Sélection des frames
- [x] Test intersection tile/frame robuste
- [x] Frames assignées à plusieurs tiles si besoin

## 🧪 Traitement par tile
- [x] Reprojection locale correcte
- [x] Empilement avec pondération
- [x] Rejet sigma/winsor/kappa OK
- [x] Tile sauvegardée dans tiles/

## 🧩 Assemblage final
- [x] Aucun appel à reproject_and_coadd
- [x] Placement direct des pixels basé sur X,Y global
- [x] Blending léger OK
- [x] Normalisation large-échelle globale OK

## 🧪 Tests multi-source
- [x] Multi-nuit → correct
- [x] Multi-site → correct
- [x] Multi-mount → correct
- [x] Multi-filtre → cohérent selon le mode choisi

## 📝 Logs
- [x] Tous les logs taggés `[GRID]`
- [x] Aucun log parasite dans le pipeline classique

