# ZeGrid — « Autoriser un secours swap limité pour la jauge » (mode d'emploi)

**Version** : ZeGrid `feat/zegrid-adaptive-resources`, 2026-11-15.
**Plateforme** : **Linux uniquement**. **État** : opt-in, désactivé par défaut.
**Ce que ça ne fait pas** : aucun changement de science, d'identité de cellule,
de magasin durable, ni de vitesse garantie.

## 1. Où se trouve l'option

Panneau **ZeGrid**, groupe **« ZeGrid run policy »**, juste sous le bloc
« Import an existing gauge artifact ». C'est une **case à cocher** :

> **« Autoriser un secours swap limité pour la jauge ; peut ralentir ; aucun crédit pour le stack »**

Sous la case, une ligne d'aide rappelle la portée réelle (« s'applique
uniquement à un run ZeGrid CPU **ACTIVE** qui calcule une jauge globale
**fraîche** ; GPU, jauge restaurée/importée et jauge par cellule restent
strictement RAM-only ; le stack, le cache, l'assemblage et l'export ne sont
jamais crédités ; décoché = comportement RAM-only historique »).

La valeur est enregistrée dans la configuration sous la clé
`zegrid_gauge_memory_policy`, avec deux valeurs seulement :
`ram_only` (défaut) et `bounded_swap`. Toute autre valeur est **refusée** par
le moteur (démarrage bloqué, pas de repli silencieux). Il n'existe **aucune**
option en ligne de commande.

## 2. Effet quand la case est cochée

1. Le moteur calcule d'abord la jauge **en RAM seule**, comme aujourd'hui.
2. Si elle ne tient pas en RAM **et** que la configuration s'y prête, il peut
   accorder un **crédit de swap borné** à **un seul** worker de jauge :
   - le crédit est plafonné à **90 %** du swap effectivement utilisable par le
     run (le plus contraignant entre le swap de l'hôte et le plafond du
     cgroup) ; **pas de plafond fixe en Gio** ;
   - le zram est compté comme RAM, pas comme disque ;
   - tout est calculé en **arithmétique entière** ; une valeur inconnue est
     traitée comme « inconnue » (refus typé), jamais comme 0 ni comme l'infini.
3. Le crédit est une **permission**, pas une réservation : le système peut
   garder les pages en RAM, évincer d'autres processus ou refuser une
   allocation. Aucun OOM n'est promis, ni dans un sens ni dans l'autre.
4. Le crédit est **fermé avant** le cache et le stack ; il n'est **jamais**
   réutilisé pour le stack, le cache, l'assemblage ou l'export, ni écrit dans
   un magasin durable.

Conséquence pratique : un run qui échouait avec « la jauge ne tient pas en RAM »
peut désormais démarrer, **plus lentement**, au lieu d'être refusé.

## 3. Quand l'option n'a aucun effet

- **Case décochée / clé absente** : comportement **RAM-only historique**,
  bit-identique (manifeste et décisions inchangés).
- **Backend GPU** : jamais crédité.
- **Jauge restaurée ou importée** : jamais créditée (elle n'est pas recalculée).
- **Jauge par cellule** (`Cell gauge mode` ≠ global) : strictement RAM-only.
- **Run hors scheduler `active`** (legacy/observe) : pas de crédit.
- **Plateforme non Linux** : la case reste **visible mais désactivée**, avec la
  raison affichée sur la même ligne ; la valeur effective du run est
  `ram_only`, et la valeur enregistrée n'est pas modifiée.

## 4. Où le voir dans les sorties du run

- Bloc `gauge_swap` du manifeste (`zegrid_manifest.json`) : **présent
  uniquement** quand la politique vaut `bounded_swap`. Il contient la
  politique, l'applicabilité et la raison, l'observation de crédit
  (`A`, `B_ram`, `S_credit`, `credit_needed`), les faits sources, les
  événements `gauge_credit_open` / `gauge_credit_closed`, et
  `execution_digest` **seulement** si un crédit a réellement été appliqué.
- `ram_contract` (diagnostics par cellule) : depuis G4, un run crédité y publie
  aussi `gauge_memory_policy`, `a_raw_bytes`, `b_ram_bytes`, `s_credit_bytes`,
  `credit_needed_bytes` et `capacity_sources` — sur **tous** les chemins
  (stream, per-cell, fallback CPU après échec GPU).
- `swap_used_bytes` reste **`null`** : c'est une permission, jamais une
  consommation annoncée. La consommation réelle se lit dans les compteurs
  observés (`/proc/vmstat`, `memory.swap.current`), séparément.

## 5. Limites à connaître avant de l'activer

- **Swap disque non garanti** : sur un hôte avec zram prioritaire, les pages
  peuvent partir en zram (RAM compressée) avant le disque. La qualification ne
  prouve pas la pagination disque.
- **Ralentissement possible** : le crédit n'achète pas de la vitesse ; il
  évite un refus.
- **Une seule jauge, un seul worker** : l'option ne finance jamais plus de
  concurrence.
- **Mode commit strict (`overcommit_memory=2`)** : non validé physiquement sur
  cet hôte (changement de sysctl interdit) ; validé par tests unitaires
  seulement.
- **Windows / GPU** : non couverts.
- **Qualification bornée** : corpus synthétique, run `< 10 min`, `n = 1` pour
  les ratios de temps, télémétrie PSI absente. Le verdict complet est
  **« qualified with limits »** — voir
  `docs/qualification/zm-gauge-swap-g3/ZM_GAUGE_SWAP_G3_CLOSURE_QUALIFICATION_20261115.md`.

## 6. Comportement par défaut : inchangé

Sans action de l'utilisateur, rien ne change :

- la case est **décochée** et la clé vaut `ram_only` ;
- la jauge doit tenir en RAM, comme avant ;
- les manifestes ne contiennent **aucun** bloc `gauge_swap` et aucun champ de
  crédit dans `ram_contract` (surface byte-identique à la version précédente) ;
- les identités de cellules, les magasins durables et la reprise ne sont pas
  affectés par la politique : celle-ci n'est **pas** une identité de cellule.
