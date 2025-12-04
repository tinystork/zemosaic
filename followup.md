# 🔁 Follow-up : Validation & Ajustements

Ceci est la liste des vérifications à effectuer sur votre implémentation.  
Cochez les cases lors des itérations :

## 🔧 Implémentation générale
- [ ] Le pipeline classique est intact (aucune différence dans les logs/classiques)
- [ ] `detect_grid_mode()` bascule proprement sans effet secondaire
- [ ] `run_grid_mode()` est complètement isolé

## 📥 Lecture du stack_plan.csv
- [ ] Fonction de parsing robuste
- [ ] Colonnes ignorées correctement
- [ ] Paths vérifiés

## 🌐 Construction de la grille
- [ ] WCS global stable
- [ ] Conversion RA/Dec → X,Y correcte
- [ ] Grille régulière générée avec overlap

## 🎛 Sélection des frames
- [ ] Test intersection tile/frame robuste
- [ ] Frames assignées à plusieurs tiles si besoin

## 🧪 Traitement par tile
- [ ] Reprojection locale correcte
- [ ] Empilement avec pondération
- [ ] Rejet sigma/winsor/kappa OK
- [ ] Tile sauvegardée dans tiles/

## 🧩 Assemblage final
- [ ] Aucun appel à reproject_and_coadd
- [ ] Placement direct des pixels basé sur X,Y global
- [ ] Blending léger OK
- [ ] Normalisation large-échelle globale OK

## 🧪 Tests multi-source
- [ ] Multi-nuit → correct
- [ ] Multi-site → correct
- [ ] Multi-mount → correct
- [ ] Multi-filtre → cohérent selon le mode choisi

## 📝 Logs
- [ ] Tous les logs taggés `[GRID]`
- [ ] Aucun log parasite dans le pipeline classique

