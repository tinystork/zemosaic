# 🔬 Suivi — Diagnostic décalage vert (Classic)

## Étapes à exécuter

- [ ] Activer le niveau `DEBUG` dans le GUI Qt
- [ ] Vérifier que ce niveau est bien propagé au logger du worker
- [ ] Lancer exactement le même dataset en :
   - mode Classic
   - mode SDS (référence saine)
- [ ] Comparer les blocs `[DBG_RGB]` dans les logs

---

## Points de comparaison clés

Comparer **strictement** :
- [ ] `P3_post_stack_core` (Classic vs SDS)
- [ ] `P4_post_merge_valid_rgb`
- [ ] `P5_pre_rgb_equalization`
- [ ] `P5_post_rgb_equalization`

---

## Hypothèse principale (à confirmer)

Une **normalisation RGB globale spécifique au mode Classic**
est appliquée **après la mosaïque**, sans tenir compte :
- du coverage
- des NaN
- du fond de ciel réel

👉 Le vert devient la référence implicite.

---

## Prochaine action (APRÈS diagnostic)

Uniquement si confirmé :
- Restreindre la stat RGB aux pixels `coverage > 0`
- ou désactiver l’égalisation globale Classic
- ou aligner Classic sur la stratégie SDS

⚠️ Aucun patch avant validation par logs.
