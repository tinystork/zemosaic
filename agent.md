Correctif “tile_infos vs pending_tiles” dans assemble_tiles()

Mission :
Corriger un bug logique dans grid_mode.py qui provoque un false negative (“Unable to read any tile for assembly”) alors que les tiles sont valides.

Contrainte absolue :
⚠️ Ne modifier que la zone autour du test if not tile_infos: qui se produit immédiatement après la première boucle for t in tiles_list: (la première occurrence dans assemble_tiles).
Ne toucher à rien d’autre dans le fichier.

🎯 Objectif exact de la modification

Dans assemble_tiles() :

Identifier le bloc suivant (exact) :

    tile_infos: list[TilePhotometryInfo] = []
    pending_tiles: list[tuple[GridTile, np.ndarray, np.ndarray, int]] = []
    ...
    for t in tiles_list:
        ...
        pending_tiles.append((t, data, mask, c))

    if not tile_infos:
        _emit(
            (
                "Unable to read any tile for assembly. "
                f"Assembly summary: attempted={len(tiles_list)}, io_fail={io_failures}, "
                f"channel_mismatch={channel_mismatches}, empty_mask={empty_masks}, kept=0"
            ),
            lvl="ERROR",
            callback=progress_callback,
        )
        return None


Remplacer le test if not tile_infos: par if not pending_tiles:
et ne rien changer d’autre dans ce bloc.

Laisser intact le second if not tile_infos: qui apparaît plus loin,
après l’harmonisation des canaux (car celui-là est correct).

✅ Résultat attendu

Après patch :

Le Grid mode ne doit plus abandonner à tort.

L’assemblage ne doit plus retourner None quand les tiles sont réellement lisibles.

Aucun autre comportement de grid_mode.py ne doit être modifié.

✔️ Checklist Codex

 Localiser la première occurrence de if not tile_infos: dans assemble_tiles().

 Vérifier qu’elle se trouve immédiatement après la boucle for t in tiles_list:.

 Remplacer uniquement cette condition par if not pending_tiles:.

 Ne rien renommer, ne rien réorganiser, ne pas toucher les imports.

 Ne pas modifier la deuxième occurrence de if not tile_infos:.

 Générer un diff propre et minimal.