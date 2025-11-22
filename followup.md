# Follow-up attendu

Merci d’inclure dans ta réponse finale :

1. [x] Comment tu as parallélisé `compute_per_tile_gains_from_coverage` (type de pool, organisation des batches).
2. [x] Comment tu as parallélisé la boucle `for ch in range(n_channels)` (CPU ou CPU+GPU hybride).
3. [x] Comment `ParallelPlan` est utilisé (rows_per_chunk, chunk_mb, cpu_workers).
4. [x] Comment les STATS_UPDATE Phase 5 ont été ajoutés et quels champs sont fournis.
5. [ ] Preuve que :
   - SDS n’a jamais été modifié,
   - les tests restent verts (pytest échoue avant collecte : `FileNotFoundError` sur la capture),
   - l’API publique n’a pas bougé.
6. [x] Comment tu t’assures que la VRAM et la RAM sont respectées pendant la seconde passe.
