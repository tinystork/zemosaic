# Guide bêta — Tuiles GPU adaptatives par cellule (expérimental)

Bêta **5.1.6**. Mission `ZM-GPU-PERCELL-PRODUCT-20261115`. Option **expérimentale, désactivée par
défaut**. Aucune ligne de commande n'est nécessaire.

## Ce que fait l'option

Quand elle est activée sur un run **ZeGrid ACTIVE** avec **GPU** :

* chaque cellule ACTIVE est planifiée sur **sa propre échelle de tuiles GPU**,
  calculée à partir de son vrai couple `(n_i, taille de patch_i)` ;
* chaque cellule démarre à **sa propre tuile admise** (mémoire hôte + VRAM) ;
* une cellule ne peut que **rétrécir** pendant le run (jamais grandir, même si
  la mémoire revient) ; une cellule profonde ne plafonne plus les suivantes ;
* les logs et le manifeste publient, par cellule : profondeur, forme du patch,
  tuile planifiée/effective, bornes hôte/device/réserve, rétrécissements et
  refus.

Désactivée, l'option laisse le plan GPU global historique **strictement
inchangé**. Elle ne change ni la science, ni la géométrie, ni le halo, ni la
gauge, ni les poids, ni le rejet, ni l'assemblage, ni les sorties FITS.

Si le GPU est désactivé ou l'ordonnanceur n'est pas `Active`, la case est
grisée et **ignorée** : la valeur enregistrée n'est pas effacée.

## Essai manuel recommandé (Windows, RTX 3070 Laptop)

1. Ouvrir ZeMosaic → mode **ZeGrid** → ordonnanceur **Active** → **GPU activé**.
2. Dans **ZeGrid run policy**, repérer la case
   **« Tuiles GPU adaptatives par cellule — expérimental »**.
3. Choisir un **dossier de sortie NEUF** (jamais un dossier existant, jamais
   `D:\M33\S50\out\fresh` ni `D:\SH2-101_mosaic_sub\out\newgauge`).
4. Lancer une première fois avec la case **décochée** (run OFF de référence),
   même corpus, mêmes réglages, autre dossier NEUF.
5. Relancer avec la case **cochée** (run ON), mêmes réglages, dossier NEUF.
6. Pour tester stop/reprise : pendant le run ON, utiliser **Arrêter et
   enregistrer la progression** ; relancer plus tard avec **Resume** (le même
   dossier de sortie et le même stockage durable). La case doit rester cochée :
   un store gelé ON est refusé explicitement si l'option est décochée.

## Journaux à conserver

Dans chaque dossier de sortie :

* `zegrid_run.log` ;
* `zegrid_manifest.json` → blocs `r55_scheduler.percell_tiles`,
  `r55_run_telemetry.gpu_pipeline` et `per_cell_diagnostics` ;
* `zemosaic_products.json` et les FITS scientifiques ;
* le dossier de stockage durable `zegrid-durable/` (si checkpoints activés).

Comparer OFF vs ON : vérifier la parité des données scientifiques FITS
(attendue bit-identique, mais **non encore qualifiée sur GPU physique**) ; les tuiles,
blocs, octets lus et temps par cellule sont les métriques explicatives.

## Limites honnêtes

* Aucune exécution GPU physique n'a été faite sur l'hôte de développement :
  `NOT_RUN`. Les gains en secondes ne sont **pas** garantis.
* La parité device↔device à tuile variable reste à qualifier sur le nœud
  Windows.
* La partition CPU a été vérifiée bit-identique (science/poids/masques/support),
  mais ce n'est pas une preuve GPU physique.

## Limites locales de validation restantes

* L'écriture du plafond avant exécution et son échec bloquant sont vérifiés par
  lecture du code. Le test de reprise injecte le plafond durable ; il ne force
  pas encore le shrink runtime complet.
* Le contrôle APPLY/RUN conserve des crédits producteur/prefetch globaux plus
  conservateurs que le plan par cellule : un refus mémoire supplémentaire reste
  possible sur une distribution très hétérogène. Aucun dépassement reproduit.
* Les tests Qt d'arrêt sont sensibles aux ressources et à l'ordre : la suite
  complète n'est pas déclarée intégralement verte.
