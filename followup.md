# followup.md — Vérifs & protocole de test (VRAM dynamique Phase 5)

## 1) Vérifs de compilation / import
- Lancer le worker sans GPU/CuPy → vérifier que l’import `zemosaic_utils` ne casse rien.
- Lancer avec CuPy → vérifier que `_probe_free_vram_bytes()` renvoie une valeur.

## 2) Vérif logs attendus
Pendant Phase 5, pour chaque canal :
- `Phase5 GPU VRAM refresh ... free_vram_mb=... budget_mb=... rows_hint=... cap_mb=...`
En cas d’OOM :
- `[GPU Reproject] OOM retry 1/3: max_chunk_mb=... rows=... free_vram_mb=...`
- puis soit succès GPU, soit `fallback CPU` (log existant) après 3/3.

## 3) Test “Upscale”
### Procédure
- Lancer Phase 5 GPU avec une VRAM initialement chargée (ex: une app GPU ouverte), puis fermer l’app après canal 1.
### Attendu
- Canal 2/3 : budget_mb (et/ou rows) augmente vs canal 1 (sans dépasser le cap user/plan).
- Temps par canal peut baisser si chunks plus gros.

## 3.bis) Test “USER cap invariant” (phase5_chunk_auto=False)
### Procédure
- Régler : `phase5_chunk_auto=False` et `phase5_chunk_mb=128` (ou 256 si tu veux).
- Lancer Phase 5 GPU avec VRAM variable entre canaux (ouvrir puis fermer une app GPU entre canal 1 et 2).

### Attendu
- Dans les logs “Phase5 GPU VRAM refresh … cap_mb=…”, `cap_mb` reflète le cap USER (≈128MB, éventuellement clampé batterie/hard_cap_vram).
- `budget_mb` (et donc la demande `max_chunk_bytes` envoyée) **ne dépasse jamais** `cap_mb`, même si `free_vram_mb` augmente entre canaux.
- En cas d’OOM, la boucle retry peut réduire sous le cap (normal), mais **aucune montée** au-dessus du cap ne doit apparaître.

## 4) Test “Downscale”
### Procédure (simple)
- Juste avant Phase 5, provoquer une pression VRAM (app GPU, ou param max_chunk_bytes très haut).
### Attendu
- Déclenchement du retry loop OOM.
- Réduction progressive des chunks.
- Si la VRAM redevient suffisante → succès GPU sans fallback CPU.

## 5) Non-régression CPU
- Exécuter un run CPU-only (use_gpu=False).
- Vérifier :
  - pas de nouveaux kwargs CPU
  - pas de différence de résultat vs avant.

## 6) Check qualité
- Vérifier qu’aucune zone noire nouvelle n’apparaît (hors non-couverture normale).
- Vérifier la présence/propagation coverage/alpha inchangée.

## 7) Nettoyage
- Si un flag temporaire a été ajouté (ex: env var pour debug), le laisser mais par défaut OFF.
- Ne pas toucher aux autres heuristiques (safe_mode_windows, clustering, intertile pairs).

## 8) Test de non-régression “hors Phase 5”
- Lancer une autre commande qui utilise reproject_and_coadd_wrapper (ex: Grid mode si pertinent) et vérifier qu’on ne voit jamais de log 'OOM retry' tant que _phase5_oom_retry n’est pas injecté.
