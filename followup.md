## 📄 `followup.md`

# Validation — SDS GPU nanpercentile fix

## 🔍 Vérifications obligatoires

### 1️⃣ Analyse du diff
- [ ] Tous les changements concernent exclusivement SDS
- [ ] Aucun fichier partagé n’a été modifié “par confort”
- [ ] Aucun appel global à cp.nanpercentile n’a été remplacé hors SDS

### 2️⃣ Test fonctionnel SDS
Lancer un run SDS avec GPU activé (dataset court accepté).

#### Logs attendus :
- [ ] ❌ ABSENCE de :
gpu_fallback_runtime_error: cupy has no attribute nanpercentile

css
Copier le code
- [ ] ❌ ABSENCE de :
Global GPU helper path failed

yaml
Copier le code
- [ ] ✅ Présence continue de la voie GPU jusqu’à la fin

### 3️⃣ Non-régression
- [ ] Mode classique : aucun changement de log ou résultat
- [ ] Mode grid : aucun changement de log ou résultat
- [ ] Aucune nouvelle warning GPU/CPU hors SDS

---

## 🧠 Rappel critique
Si un changement améliore “globalement” le code mais touche une autre voie que SDS,
alors **la mission est considérée comme échouée**, même si le bug disparaît.

Le but est :
👉 **corriger SDS**
👉 **ne rien casser**
👉 **ne rien embellir ailleurs**