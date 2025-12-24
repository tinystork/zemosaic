# Follow-up — Validation & critères d’acceptation

## Tests manuels
1) Windows, dataset qui hangait (pairs > 4000, preview=512)
   - Lancer mosaïque avec photometric intertile ON
   - Attendu:
     - Le log montre "[Intertile] SAFE_MODE..." OU bien lock activé
     - Le log affiche des "progress pairs_done=..." régulièrement
     - Le traitement arrive à "Processing completed successfully." (ou au moins dépasse le point 348/4342)

2) Dataset petit (pairs < 1000)
   - Attendu: comportement inchangé, parallélisme conservé.

## Critères d’acceptation
- Plus de hang silencieux reproductible à ~348/4342 sur le dataset fourni.
- En cas de nouveau hang/crash: présence d’un fichier faulthandler_intertile.log exploitable.
- Aucun changement d’UI, aucune régression sur le reste du pipeline.

## Notes d’implémentation
- Si lock choisi: le lock doit entourer *uniquement* les appels reproject_interp (et éventuellement WCS build),
  pas toute la fonction, pour limiter l’impact perf.
- Si safe_mode mono-worker choisi: seuils conservateurs (pairs>=2000/4000) + seulement Windows.
