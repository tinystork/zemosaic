# Qualification courte — découplage workers counts/pairs

Mission `ZM-GAUGE-COUNTS-DECOUPLING-20261115`, phase implementation.
Baseline : `feat/zegrid-adaptive-resources` @ `df49867dba0e595401ee4d80e29f71cc484e74da`.

## Ce que ce témoin prouve (Linux/TinyDebian uniquement)

- même corpus FITS local réel (M106, 6 frames décodées par le décodeur produit) ;
- même configuration/science pour toutes les armes ; seuls `workers` et
  `counts_workers` changent ;
- payload scientifique **bit-à-bit** identique à l'arme OFF (référence,
  exclusions, coefficients, poids, masques actifs, ids) ;
- les diagnostics existants `phases.counts.seconds` / `phases.pairs.seconds`
  mesurent les temps sous-phases ; le pool counts est réellement `process`
  (aucun fallback série caché) ;
- le probe `spawn` Linux confirme la propagation du kwarg sous un contexte
  spawn.

## Ce que ce témoin NE prouve PAS

- **aucune** qualification Windows physique : pics commit privé parent+enfants,
  working-set agrégé, `spawn` Windows et débit disque restent non mesurés ;
- aucun gain de run M33 : la mesure porte sur 6 frames locales, pas sur
  l'admission M33 (Fc/Fp du modèle et Nc=3 M33 ne sont pas exercés) ;
- pas de gros bench ni de stress RAM ; une répétition bornée ne retire pas le
  bruit d'ordonnancement Linux.

## Reproduire

```bash
python docs/qualification/zm-gauge-counts-decoupling-20261115/run_ab.py \
  --frames 6 --reps 2 \
  --out docs/qualification/zm-gauge-counts-decoupling-20261115/ab_results.json
```

`ab_pilot.json` est le premier passage (1 répétition) ; `ab_results.json` est le
résultat borné retenu dans le rapport de mission.

Rework-1 n'a modifié ni le noyau photométrique, ni les pools, ni l'admission :
les mesures ci-dessus restent valides et n'ont pas été relancées. Seuls les
diagnostics (synchronisation du pic séquentiel après shrink Np) et le câblage
option/GUI ont changé.

## Activation bêta 5.1.4 (2026-11-15)

À la demande de Tristan, le découplage est désormais **activé par défaut**, sans
nouveau contrôle GUI. Une configuration existante dépourvue de la clé bénéficie
automatiquement du nouveau défaut après mise à jour et redémarrage. Un `false`
explicitement persisté reste respecté ; `zegrid_gauge_counts_decoupling: false`
dans le JSON permet le retour au comportement historique.

Le scheduler doit être `active` (défaut produit). Il dimensionne automatiquement
les workers counts et pairs séparément selon les limites CPU et les empreintes
mémoire, sous le même budget et la même marge de 15 %. Les pools sont séquentiels ;
le coût est `max(Nc*Fc, Np*Fp)`, pas leur somme. Aucun nombre fixe de workers n'est
imposé. Le crédit swap, lorsqu'il est appliqué, maintient les deux nombres à 1.
Une jauge restaurée n'exécute pas de nouveau pool counts.

La bêta permet l'essai Windows par l'utilisateur ; la qualification physique
Windows reste à faire. La publication ne met pas à jour le runtime Windows et
ne modifie pas le run M33 déjà lancé. Les mesures ci-dessus restent Linux only.
