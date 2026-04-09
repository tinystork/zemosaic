# RUN C — Seams root-cause instrumentation checklist

Objectif principal: isoler l’origine des seams sur pipeline réel (pas du tuning cosmétique).
Objectif secondaire: capturer les indices de thrashing sans détourner la mission.

## 1) Appliquer le profil d’overrides (sans écraser aveuglément)

```bash
cd /home/tristan/zemosaic/zemosaic
cp -a zemosaic_config.json zemosaic_config.backup.before_run_c.json
python3 - <<'PY'
import json
base='zemosaic_config.json'
over='profiles/run_c_seams_instrumented_overrides.json'
with open(base) as f: cfg=json.load(f)
with open(over) as f: ov=json.load(f)
cfg.update(ov)
with open(base,'w') as f: json.dump(cfg,f,indent=4)
print('Applied overrides:', len(ov))
PY
```

## 2) Sanity check des clés critiques

```bash
python3 - <<'PY'
import json
cfg=json.load(open('/home/tristan/zemosaic/zemosaic/zemosaic_config.json'))
keys=[
 'output_dir','logging_level','enable_resource_telemetry','two_pass_coverage_renorm',
 'intertile_photometric_match','intertile_prune_k','intertile_prune_weight_mode',
 'enable_tile_weighting','tile_weight_v4_enabled','cache_retention','cleanup_temp_artifacts'
]
for k in keys:
    print(f"{k}: {cfg.get(k)}")
PY
```

## 3) Lancer le run via GUI (contexte habituel)

- Dataset: `/media/tristan/X10 Pro/mosaic/andromeda/out/EQ_CLASSIC/zemosaic_temp_master_tiles`
- Output attendu: `/media/tristan/X10 Pro/mosaic/andromeda/out/V4 RUN C SEAMS INSTRUMENTED`

## 4) Contrôles post-run immédiats (preuve pipeline réel)

```bash
LOG='/media/tristan/X10 Pro/mosaic/andromeda/out/V4 RUN C SEAMS INSTRUMENTED/zemosaic_worker.log'

grep -n 'Pair pruning summary' "$LOG"
grep -n 'assemble_info_intertile_photometric_applied' "$LOG"
grep -n 'apply_photometric_summary' "$LOG"
grep -n '\[Phase5\] tile_weights summary' "$LOG"
grep -n '\[TwoPass\] run_second_pass_coverage_renorm start' "$LOG"
grep -n '\[TwoPass\] Computed gains count=' "$LOG"
grep -n 'run_success_processing_completed' "$LOG"
```

## 5) Lecture orientée seams (ce qu’on veut conclure)

- Si `Pair pruning summary` montre un pruning fort (raw >> pruned), suspect #1 = graphe retenu trop maigre.
- Si `tile_weights summary` montre une dynamique extrême, suspect #2 = weighting dominateur.
- Si `apply_photometric_summary` montre des corrections fortes/instables, suspect #3 = homogénéisation insuffisante.
- Si ces trois signaux coexistent, on tient le combo root-cause le plus probable des seams.

## 6) Side-check thrashing (secondaire)

```bash
python3 - <<'PY'
import csv, statistics
p='/media/tristan/X10 Pro/mosaic/andromeda/out/V4 RUN C SEAMS INSTRUMENTED/resource_telemetry.csv'
rows=list(csv.DictReader(open(p)))
for k in ['ram_used_mb','ram_available_mb','gpu_used_mb','gpu_free_mb','cpu_percent']:
    vals=[]
    for r in rows:
        try: vals.append(float(r[k]))
        except: pass
    if vals:
        print(k, 'min',min(vals),'med',statistics.median(vals),'max',max(vals))
PY
```

But: observer la pression mémoire pendant Two-Pass gains, sans perdre le focus seams.
