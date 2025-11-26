✅ agent.md (prêt pour Codex High / Max)

Mission : Corriger la régression d’alignement introduite dans la branche V4WIP

🎯 Objectif

Corriger la déformation géométrique observée dans la branche V4WIP lors de l’empilement classique.
La cause est en amont de la voie GPU : elle provient de la phase d’alignement CPU, plus précisément de la fonction align_images_in_group dans le fichier zemosaic_align_stack.py.

Le GPU empile simplement les images alignées : ne rien modifier dans la voie GPU.

📌 Contexte + Hypothèse principale

Dans V4WIP, les images présentent une déformation ou un étirement qui n’existait pas en V4.2.5.

La pipeline d’alignement CPU fonctionne en deux étapes :

pré-alignement par FFT / corrélation de phase

alignement fin via astroalign.register(source, target) (transformation affine)

La déformation est générée dans cette étape CPU, jamais dans la voie GPU.

On suspecte :

soit une modification du pré-alignement FFT (mauvais décalage appliqué en amont)

soit une déviation des données d'entrée envoyées à astroalign

soit une erreur dans l’usage ou l’interprétation de la transformation retournée par astroalign

Mais pas un bug GPU.

📁 Fichiers à modifier (et uniquement ceux-là)

zemosaic_align_stack.py
👉 fonction : align_images_in_group

🚫 Ne pas modifier :

zemosaic_align_stack_gpu.py

aucune fonction GPU, aucun kernel, aucune logique de coadd.

💡 Tâches à effectuer
1. Isoler et analyser l’étape fautive

Instrumenter align_images_in_group pour comparer :

le résultat FFT (dx, dy, conf)

les entrées source et target envoyées à astroalign

la transformation affine retournée par astroalign.register()

2. Comparer contre V4.2.5

Pour un même jeu d’images, logguer :

les images “pré-alignées FFT”

la transformation affine retournée

l’image alignée finale

3. Créer un mode “FFT-only” (temporaire, activable par flag interne)

Permettra de tester :

skip_astroalign = True


Si la déformation disparaît → la cause est bien dans la seconde étape.

⚠️ Ce flag doit être temporaire et non exposé à l’utilisateur.

4. Corriger la cause

Selon ce que les logs révéleront :

mauvais couple source/target ?

décalage FFT appliqué deux fois ?

cropping / normalisation incorrecte ?

mauvaise orientation / dtype incompatible ?

application erronée de la matrice de transformation ?

L’objectif est de restaurer le comportement exact de V4.2.5.

5. Robustifier l'appel à astroalign

Vérifier input shapes (H, W) vs (H, W, 3)

S'assurer que la normalisation en float32 est identique à V4.2.5

Forcer éventuellement un “sanity check” sur la matrice retournée

6. Ne pas toucher à la partie GPU

Le GPU reçoit des images déjà warpées.
Toute déformation provient avant l’empilement.

✔️ Validation

Le correctif est réussi si :

un jeu d’images problématique en V4WIP reprend la forme correcte obtenue en V4.2.5

aucune transformation affine aberrante n’apparaît dans les logs

le pipeline CPU reste compatible avec la voie GPU existante

aucun changement ne touche SDS ou GPU

➤ Livrables attendus

Patchs sur zemosaic_align_stack.py

Petits logs de debug pour comparaison V4.2.5 / V4WIP

Aucun changement GPU