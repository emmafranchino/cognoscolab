
# File
**Numero di File**:   
1° anno = 49  
2° anno = 70  
3° anno = 52 

**Totale** = 171

**File totali analizzati = 169**

## File mancanti

File mancanti:  
1° anno → 045   
2° anno → LC_019  
3° anno → LL_010

| ID             | Motivo                                                                                             | Outcome |
|----------------|----------------------------------------------------------------------------------------------------|---------|
| **042**        | Lo script non riesce a leggere la parte delle associazioni libere. Le associazioni erano infatti riportate tutte in un'unica cella. Le ho trasportate manualmente in celle diverse, ma non riesce comunque a leggerle. Non capisco il perché. Dunque, per evitare di far rompere lo script. | Incluso dopo averlo formattato correttamente. |
| **045**        | Presenta tutti i fogli con nomi diversi e i dati riportati in modo sparso. | Escluso per i troppi errori. |
| **LL_010**     | Manca proprio dalle cartelle condivise. Probabilmente è stato un errore di battitura nell'indicizzazione o è andato perso.| Non presente. |
| **LC_019** |  LC_016 e LC_019 sono identici, partecipante ripetuto. | Escluso|
| **LC_064_incompleto** | Solamente incompleto per quanto riguarda il task *'breve_testo'*, il resto é presente. | Incluso (solo una parte mancante). |

**Totale:** 171 − 3 = **168**

# Dati mancanti
## Task associazioni libere e valenza

**Parole mancanti nei file analizzati**:  
I file analizzati presentano alcuni dati mancanti, soprattutto per quanto riguarda il task delle associazioni e il task della valenza --> alcune 'cue word' sono mancanti o ripetute più volte all'interno del task. 

→ per tenere traccia delle mancanze di queste parole ho creato due file .txt:
1. per il task associativo = 'extracted_data_2024/missing_keywords_associations_task.txt'
2. per il task della valenza = 'extracted_data_2024/missing_keywords_valence_task.txt'

### Task associazioni libere
In tutti i file manca la parola chiave *‘studente’* e nei file la parola *‘biologia’* é assente e al suo posto la parola *‘psicologia’* é ripetuta 2 volte:
- 011.xlsx
- 012.xlsx
- 013.xlsx 
- 015.xlsx
- 022.xlsx
- 026.xlsx
- 037.xlsx
- 039.xlsx
- 040.xlsx
- 046.xlsx

→ tutti i file hanno una lunghezza di 40 parole, invece di 41. 
Lista dei file con parole mancanti (tutti quelli del primo anno, tranne il 45, che é stato escluso) → 49-1 = 48

Quindi se fosse stato tutto giusto la lunghezza totale del df sarebbe 41 * 168 = 6.888

Ma dato che c’é questo problema, si ha:(40 * 48)+ (41 * 120) = 6.840

### Task valenza
Le parole *'psicologia'* e '*biologia'* per tutti i partecipanti elencati sopra (tranne l'ultimo 046) non sono presenti nel task della valenza, ma solo in quello associativo.  

→ Parole totali che ci dovrebbero essere - 1 mancante (*'studente'*) per ogni partecipante primo anno (48) - 2 parole per 9 partecipanti in cui ci sono stati i problemi (18):

(40 * 48) + (41 * 120) - 18 = 6.822

**!!** Nel file 'extracted_data_2024/associations_valence_2024.csv' si trovano tutti i dati raggruppati (sia le associazioni che la valenza di ogni cue word). Sono state inserite anche le parole presenti nel task associativo e non nel task della valenza (quindi in tutti i campi della valenza si avranno valori nulli) **!!**

## Task questionari e breve testo
Per quanto riguarda invece i dati mancanti nei questionari o nella sezione *'breve_testo'* si possono vedere i dati mancanti e il file dal quale mancano (quindi il partecipante) direttamente nel file dove vengono riportati tutti i dati estratti ('extracted_data_2024/text_task_2024.csv' e 'extracted_data_2024/questionnaires_task_2024.csv')

### Questionari
Ci sono dati mancanti, poiché non compilati dai partecipanti. A volte mancano anche genere ed età. 

### Breve testo
Questa sezione non é stata completata da alcuni partecipanti, principalmente nei partecipanti del primo anno, dove sono presenti solo 16 testi su 49 partecipanti. Anche alcuni di altri anni come LC_064_incompleto. Si possono visualizzare i file vuoti nel file csv dove vengono riportati sia i dati che i nomi dei file. 