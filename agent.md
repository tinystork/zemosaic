# Agent.md – Nettoyage contrôlé de l’historique Git entre `d5c29f1` et `accbc75`

## Contexte

Branche concernée : `dumbcodex` (puis éventuel report sur `V4WIP`).

Cette branche contient une série de commits expérimentaux entre :

- `d5c29f1` – `update`  (**à conserver**)
- `accbc75` – `world2px error 38c876ae` (**à conserver**)
- `f88d19e` – `world2px error fix in 38c876ae continued` (**à conserver, HEAD cible**)

Tous les commits **intermédiaires** entre `d5c29f1` et `accbc75` sont des essais / bidouilles qu’on souhaite **supprimer de l’historique**, tout en gardant l’état de code correspondant à la chaîne :

```text
d5c29f1  update
…        (commits expérimentaux à supprimer)
accbc75  world2px error 38c876ae
f88d19e  world2px error fix in 38c876ae continued  (HEAD)
L’objectif est de réécrire l’historique pour obtenir :

text
Copier le code
f88d19e  world2px error fix in 38c876ae continued  (HEAD)
accbc75  world2px error 38c876ae
d5c29f1  update
<commits plus anciens inchangés>
Mission
Produire un plan de réécriture d’historique Git simple et sûr qui :

 Conserve exactement les commits suivants : d5c29f1, accbc75, f88d19e.

 Supprime tous les commits situés entre d5c29f1 et accbc75 dans la branche.

 Ne modifie pas le contenu final des fichiers par rapport à ce que donne actuellement f88d19e (seule la forme de l’historique change).

 Fonctionne avec Git en ligne de commande (PowerShell) et éventuellement via les outils Git intégrés de VS Code (rebase interactif).

 Prévoit une branche de sauvegarde avant toute réécriture d’historique.

Important : on est sur une branche de travail, la réécriture de l’historique est acceptable ; il faudra simplement forcer le push si la branche existe déjà sur origin.

Contraintes
 Ne pas toucher au contenu du code pour cette mission (aucune modification de fichiers : uniquement de l’historique Git).

 Ne pas modifier les tags existants, à moins que ce soit explicitement demandé (ce n’est pas le cas ici).

 Fournir les commandes sous une forme directement copiable sous Windows (PowerShell).

 Expliquer clairement à quel moment un git push --force-with-lease est nécessaire, et pourquoi.

Ressources disponibles
Le dépôt local contient déjà la branche dumbcodex et la branche V4WIP.

L’utilisateur est à l’aise avec Git en ligne de commande, mais a rencontré des blocages avec des états “needs merge” / MERGE_HEAD missing. On veut donc un scénario propre, facile à suivre.

Attendu
À la fin de la mission, l’agent doit fournir :

Une procédure détaillée (pas à pas) pour :

créer une branche de backup,

lancer un rebase interactif au bon endroit,

marquer les commits à supprimer,

finaliser le rebase,

vérifier le résultat avec git log.

Les commandes Git exactes à exécuter (PowerShell / CLI).

Les instructions pour mettre à jour la branche distante (origin/dumbcodex) via git push --force-with-lease en expliquant les implications.

L’historique final souhaité est :

f88d19e  world2px error fix in 38c876ae continued
accbc75  world2px error 38c876ae
d5c29f1  update
sur la branche dumbcodex (ou sur la branche explicitement ciblée dans la procédure).