"""Generate GPT-OSS personas matched one-to-one to the Rassereno human sample.

Every human in the final sample (psychology Sets 4 + 5b, high school Set 5a, physics Set 5a)
gets one simulated persona, which receives exactly the cue list its human counterpart got and a
profile copied from that human wherever the data has it (gender, age, year of study, programme).
On top of that, ~15% spare personas are generated per (sample, cue variant) stratum, so that
personas whose answers still fail validation after all retries can be dropped.

Each persona completes, in two separate calls (as in the original 2024 script):
    1. the free association + valence task on its cue list (cue order shuffled per persona);
    2. the reduced 18-item MAS-IT, the version the humans were scored on.

Workflow (the second step can run on a different machine, with only the manifest copied over):

    # 1. build the persona manifest from the human data (local, needs the repo data + pandas)
    python src/gpt_oss_data_generation.py personas --out rassereno/data/gpt_oss_2026

    # 2. query the model (needs `openai`; resumable: finished personas are skipped)
    python src/gpt_oss_data_generation.py generate --out rassereno/data/gpt_oss_2026 \
        --base-url http://localhost:8000/v1 --model openai/gpt-oss-20b --workers 8

    # 3. (re)build the CSVs from the raw responses; `generate` already does this at the end
    python src/gpt_oss_data_generation.py collect --out rassereno/data/gpt_oss_2026

The data folder (--out) holds what the run produces: personas.csv, raw/ (one JSON per persona) and
generation_report.csv. The answers, in the formatted-CSV schema used for the humans
(associations_valence_gpt_oss.csv, mas_it_gpt_oss.csv), go to out/formatting/<same folder name>.
"""

import argparse
import csv
import json
import math
import os
import random
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from config import RESSERENO_ROOT

FORMATTING = RESSERENO_ROOT / "out" / "formatting"

SEED = 2026
OVERSAMPLE = 0.15
MAX_RETRIES = 3

# --- cue sets, spelled as in the human association files ------------------------------------

SET_4 = [
    "ansia", "benessere", "biologia", "cognizione", "compito", "comportamento", "conoscenza",
    "creatività", "curiosità", "emozione", "equazione", "esame", "esercitazione", "fisica",
    "frustrazione", "futuro", "grafico", "insegnante", "laurea", "lavoro", "matematica", "mente",
    "modello", "neuroscienze", "noia", "numeri", "personalità", "professore", "psicologia",
    "relazione", "scienza", "scoperta", "scuola", "sfida", "statistica", "stress", "studente",
    "tecnologia", "terapia", "università", "voto",
]  # fmt: skip

# the 40 cues shared by every Set 5 variant
SET_5_CORE = [
    "analisi", "ansia", "arte", "benessere", "compagni", "conoscenza", "creatività", "dati",
    "docente", "equazione", "esame", "esperimento", "fallimento", "fisica", "futuro", "genitori",
    "imbarazzo", "informatica", "innovazione", "ipotesi", "laboratorio", "logica", "matematica",
    "numeri", "panico", "paura", "problema", "programmazione", "regole", "ricerca", "scienza",
    "scienziata", "scienziato", "scoperta", "scuola", "sfida", "statistica", "stress", "vita",
    "voto",
]  # fmt: skip

CUE_SETS = {
    # first-year psychology students never got "studente". 10 of them also lost "biologia" to a
    # form error (a duplicated "psicologia"); their personas get the intended list.
    "4_first_year": [w for w in SET_4 if w != "studente"],
    "4": SET_4,
    "5a_meditazione": SET_5_CORE + ["intelligenza_artificiale", "meditazione"],
    "5a_chatgpt": SET_5_CORE + ["intelligenza_artificiale", "chatgpt"],
    "5b": SET_5_CORE + ["counselling", "meditazione"],
}

# how a cue is written in the prompt, when different from its file spelling
CUE_DISPLAY = {"intelligenza_artificiale": "intelligenza artificiale", "chatgpt": "ChatGPT"}

# --- MAS-IT, reduced form: item text in the Qualtrics order, original 23-item numbering -------

MAS_IT_ITEMS = [
    (1, "Avere qualcuno che mi guarda moltiplicare 12x23 su carta"),
    (3, "Sostenere un esame di matematica"),
    (4, "Se mi viene chiesto davanti a molte altre persone di calcolare EUR 9,36 diviso per 4"),
    (5, "Calcolare una serie di moltiplicazioni su carta"),
    (6, "Dover affrontare un test di matematica a sorpresa in una classe"),
    (7, "Dover memorizzare una tabellina"),
    (8, "Se mi viene chiesto di calcolare i 3/5 di una percentuale"),
    (9, "Se mi viene chiesta una domanda di matematica da un/una insegnante di fronte alla classe"),
    (11, "Se mi viene chiesto di sommare il numero di persone in una stanza"),
    (12, "Calcolare quanti giorni mancano al compleanno di una persona"),
    (13, "Ricevere un numero di telefono e doverlo ricordare"),
    (14, "Calcolare quanto tempo mi rimane prima di partire per il lavoro o il luogo di studio"),
    (18, 'Leggere la parola "algebra"'),
    (19, "Ascoltare qualcuno che parla di matematica"),
    (20, "Leggere un testo di matematica"),
    (21, "Guardare qualcuno risolvere un problema di algebra"),
    (22, "Frequentare una lezione di matematica"),
    (23, "Guardare un/una insegnante scrivere equazioni alla lavagna"),
]

# --- persona options for what the human data does not record ---------------------------------

SOCIOECONOMIC = ["basse", "medie", "alte"]
AGE_RANGE = {"high_school": (17, 19), "university": (19, 25)}

# human id prefix -> (sample, programme as completed in "Sei iscritt{o/a} {programme}")
HUMAN_GROUPS = {
    "first_year": ("psychology", "al 1° anno del corso di laurea triennale in psicologia"),
    "second_year": ("psychology", "al 2° anno del corso di laurea triennale in psicologia"),
    "third_year": ("psychology", "al 3° anno del corso di laurea triennale in psicologia"),
    "interfacce": ("psychology", "al corso di laurea triennale in psicologia"),
    "liceo_sci": ("high_school", "al liceo scientifico"),
    "liceo_lin": ("high_school", "al liceo linguistico"),
    "liceo": ("high_school", "alla scuola superiore"),
    "fisica_lecce": ("physics", "al corso di laurea in fisica"),
    "fisica_stem": ("physics", "al corso di laurea in fisica"),
    "fisica": ("physics", "al corso di laurea in fisica"),
}
# left out of the human sample: on hold, see merge_and_split_datasets.ipynb
EXCLUDED_PREFIXES = ("fisica_unitn", "teacher", "altro")

# --- prompts -----------------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "{persona}\n\n"
    "Rispondi sempre e solo nel formato richiesto, senza titoli, spiegazioni o commenti."
)

ASSOCIATION_TASK = """\
Per ciascuna delle seguenti {n} parole chiave:
1. valuta la parola chiave su una scala da 1 (non mi piace per nulla) a 5 (mi piace molto);
2. scrivi fino a tre parole o espressioni che ti vengono in mente pensando alla parola chiave \
(libere associazioni). Se te ne vengono in mente meno di tre, lascia vuoti i campi mancanti;
3. valuta ciascuna associazione sulla stessa scala da 1 a 5.

Parole chiave: {cues}.

Restituisci esattamente {n} righe, una per parola chiave e nello stesso ordine, ciascuna nel \
formato:
parola_chiave,valenza,associazione_1,valenza_1,associazione_2,valenza_2,associazione_3,valenza_3

Non usare virgole all'interno delle associazioni. Se lasci un'associazione vuota, lascia vuota \
anche la sua valenza, mantenendo le virgole. Esempio di riga con due sole associazioni:
parola_chiave,valenza,associazione_1,valenza_1,associazione_2,valenza_2,,"""

MAS_IT_TASK = """\
Per ciascuna delle seguenti situazioni, indica quanto ti sentiresti ansios{suffix} su una scala \
da 1 (per niente ansios{suffix}) a 5 (molto ansios{suffix}).

{items}

Restituisci una sola riga con i {n} punteggi, nell'ordine delle situazioni e separati da \
virgole, nel formato:
punteggio_1,punteggio_2,...,punteggio_{n}"""


@dataclass
class Persona:
    persona_id: str
    human_id: str
    sample: str
    cue_variant: str
    is_spare: bool
    gender: str  # "F" / "M"
    gender_source: str  # "human" / "random"
    age: int
    age_source: str
    programme: str
    socioeconomic: str
    cue_order: str  # "|"-joined, already shuffled

    @property
    def cues(self):
        return self.cue_order.split("|")

    @property
    def suffix(self):
        return "a" if self.gender == "F" else "o"

    def description(self):
        s = self.suffix
        student = "una studentessa" if self.gender == "F" else "uno studente"
        return (
            f"Sei {student} italian{s} di {self.age} anni. Sei iscritt{s} {self.programme}. "
            f"Sei cresciut{s} e vivi in condizioni socio-economiche {self.socioeconomic}. "
            "Ricorda che le tue risposte devono essere originali, creative e coerenti con le tue "
            "caratteristiche uniche."
        )


# ================================================================================================
# 1. persona manifest
# ================================================================================================


def id_prefix(participant_id):
    """Longest HUMAN_GROUPS prefix of an id ("fisica_lecce_003" -> "fisica_lecce")."""
    matches = [p for p in HUMAN_GROUPS if participant_id.startswith(p + "_")]
    return max(matches, key=len) if matches else None


def load_humans():
    """One row per human in the final sample: id, group prefix, own cue words, gender, age."""
    import pandas as pd

    assoc = pd.concat(
        [
            pd.read_csv(FORMATTING / "psychology_BA_2024" / "associations_valence_2024.csv"),
            pd.read_csv(
                FORMATTING / "all_students_2025" / "associations_valence_task_qualtrics.csv"
            ),
        ]
    )
    assoc = assoc[~assoc["participant_id"].str.startswith(EXCLUDED_PREFIXES)]
    humans = assoc.groupby("participant_id")["cue_word"].agg(set).rename("cues").reset_index()
    humans["prefix"] = humans["participant_id"].map(id_prefix)
    unknown = humans.loc[humans["prefix"].isna(), "participant_id"].tolist()
    if unknown:
        raise ValueError(f"participants with no known group: {unknown}")

    q_2024 = pd.read_csv(FORMATTING / "psychology_BA_2024" / "all_questionnaires_task_2024.csv")
    q_2025 = pd.read_csv(FORMATTING / "all_students_2025" / "all_questionnaires_2025.csv")
    q_2024["gender"] = q_2024["gender"].map({"Femminile": "F", "Maschile": "M"})
    info = pd.concat(
        [q_2024[["participant_id", "gender", "age"]], q_2025[["participant_id", "gender"]]]
    )
    return humans.merge(info, on="participant_id", how="left")


def cue_variant(prefix, cues):
    if prefix == "first_year":
        return "4_first_year"
    if prefix in ("second_year", "third_year"):
        return "4"
    if prefix == "interfacce":
        return "5b"
    # high school and physics: Set 5a, with ChatGPT replacing meditazione from 2026-02-27
    return "5a_chatgpt" if "chatgpt" in cues else "5a_meditazione"


def make_persona(human, persona_id, is_spare, rng):
    sample, programme = HUMAN_GROUPS[human.prefix]
    gender, age = human.gender, human.age
    gender_source = "human" if gender in ("F", "M") else "random"
    if gender_source == "random":
        gender = rng.choice(["F", "M"])
    age_source = "human" if age == age else "random"  # NaN != NaN
    if age_source == "random":
        age = rng.randint(*AGE_RANGE["high_school" if sample == "high_school" else "university"])

    variant = cue_variant(human.prefix, human.cues)
    cues = CUE_SETS[variant][:]
    rng.shuffle(cues)
    return Persona(
        persona_id=persona_id,
        human_id=human.participant_id,
        sample=sample,
        cue_variant=variant,
        is_spare=is_spare,
        gender=gender,
        gender_source=gender_source,
        age=int(age),
        age_source=age_source,
        programme=programme,
        socioeconomic=rng.choice(SOCIOECONOMIC),
        cue_order="|".join(cues),
    )


def build_personas(out_dir, seed, oversample):
    rng = random.Random(seed)
    humans = load_humans()
    humans["sample"] = humans["prefix"].map(lambda p: HUMAN_GROUPS[p][0])
    humans["variant"] = [cue_variant(p, c) for p, c in zip(humans["prefix"], humans["cues"])]

    personas = [
        make_persona(h, f"gpt_oss_{h.participant_id}", False, rng)
        for h in humans.itertuples()
    ]
    # spares: an extra persona for a random ~oversample share of the humans in each stratum
    for _, stratum in humans.groupby(["sample", "variant"]):
        n_spare = math.ceil(oversample * len(stratum))
        spare_rows = stratum.sample(n=n_spare, random_state=rng.randrange(2**32))
        personas += [
            make_persona(h, f"gpt_oss_{h.participant_id}_spare", True, rng)
            for h in spare_rows.itertuples()
        ]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "personas.csv", [asdict(p) for p in personas])

    print(f"{len(personas)} personas ({sum(p.is_spare for p in personas)} spares):")
    counts = {}
    for p in personas:
        key = (p.sample, p.cue_variant, "spare" if p.is_spare else "matched")
        counts[key] = counts.get(key, 0) + 1
    for key, n in sorted(counts.items()):
        print(f"  {'  '.join(key):45s} {n}")


def read_personas(out_dir):
    with open(out_dir / "personas.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["is_spare"] = r["is_spare"] == "True"
        r["age"] = int(r["age"])
    return [Persona(**r) for r in rows]


# ================================================================================================
# 2. prompts and response validation
# ================================================================================================


def normalise(word):
    """Lowercase, no accents, spaces for underscores: how cues are matched in the answers."""
    word = unicodedata.normalize("NFKD", word.strip().strip("*").lower())
    word = "".join(c for c in word if not unicodedata.combining(c))
    return re.sub(r"[\s_]+", " ", word)


def association_prompt(persona):
    shown = [CUE_DISPLAY.get(c, c) for c in persona.cues]
    return ASSOCIATION_TASK.format(n=len(shown), cues=", ".join(shown))


def mas_it_prompt(persona):
    items = "\n".join(f"{i}. {text}" for i, (_, text) in enumerate(MAS_IT_ITEMS, start=1))
    return MAS_IT_TASK.format(suffix=persona.suffix, items=items, n=len(MAS_IT_ITEMS))


def parse_rating(value):
    value = value.strip()
    return int(value) if value in {"1", "2", "3", "4", "5"} else None


def parse_associations(text, cues):
    """Parse the association task answer.

    Returns (rows, errors): one dict per cue in the human-file schema, and a list of problems.
    The answer is valid only if errors is empty.
    """
    by_norm = {normalise(c): c for c in cues}
    rows, errors = {}, []

    for line in text.splitlines():
        line = line.strip().strip("`")
        if "," not in line:
            continue  # blank lines, fences, stray prose
        fields = [f.strip() for f in line.split(",")]
        cue = by_norm.get(normalise(fields[0]))
        if cue is None:
            errors.append(f"unknown cue: {line[:60]!r}")
            continue
        if cue in rows:
            errors.append(f"duplicate cue: {cue}")
            continue

        fields += [""] * (8 - len(fields))
        if len(fields) > 8 and any(fields[8:]):
            errors.append(f"{cue}: more than 8 fields")
        row = {"cue_word": cue, "valence_cue_word": parse_rating(fields[1])}
        if row["valence_cue_word"] is None:
            errors.append(f"{cue}: invalid cue valence {fields[1]!r}")
        for k in range(1, 4):
            word, rating = fields[2 * k], fields[2 * k + 1]
            row[f"association_{k}"] = word.lower() or None
            row[f"valence_association_{k}"] = parse_rating(rating) if word else None
            if word and row[f"valence_association_{k}"] is None:
                errors.append(f"{cue}: invalid valence {rating!r} for {word!r}")
            elif not word and rating:
                errors.append(f"{cue}: valence without association {k}")
        rows[cue] = row

    missing = [c for c in cues if c not in rows]
    if missing:
        errors.append(f"missing cues: {missing}")
    return [rows[c] for c in cues if c in rows], errors


def parse_mas_it(text):
    """Return (scores, errors): the last line holding exactly the expected number of 1-5 ratings."""
    n = len(MAS_IT_ITEMS)
    for line in reversed(text.splitlines()):
        values = [v for v in re.split(r"[,\s;]+", line.strip().strip("`")) if v]
        scores = [parse_rating(v) for v in values]
        if len(scores) == n and None not in scores:
            return scores, []
    return None, [f"no line with {n} ratings between 1 and 5"]


# ================================================================================================
# 3. generation
# ================================================================================================


def ask(client, args, persona, prompt):
    kwargs = {}
    if args.reasoning_effort:
        kwargs["reasoning_effort"] = args.reasoning_effort
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(persona=persona.description())},
            {"role": "user", "content": prompt},
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        **kwargs,
    )
    choice = response.choices[0]
    return choice.message.content or "", choice.finish_reason


def run_task(client, args, persona, prompt, parse):
    """Ask until the answer parses, at most 1 + max_retries times. Every attempt is kept."""
    attempts = []
    for _ in range(1 + args.max_retries):
        try:
            text, finish = ask(client, args, persona, prompt)
        except Exception as e:  # network errors, server restarts: count as a failed attempt
            attempts.append({"response": None, "errors": [f"request failed: {e}"]})
            time.sleep(5)
            continue
        parsed, errors = parse(text)
        if finish == "length":
            errors = errors + ["truncated: max_tokens reached"]
        attempts.append({"response": text, "finish_reason": finish, "errors": errors})
        if not errors:
            return parsed, attempts
    return None, attempts


def simulate(client, args, persona, raw_dir):
    start = time.time()
    assoc_prompt = association_prompt(persona)
    assoc, assoc_attempts = run_task(
        client, args, persona, assoc_prompt, lambda t: parse_associations(t, persona.cues)
    )
    mas_prompt = mas_it_prompt(persona)
    mas, mas_attempts = run_task(client, args, persona, mas_prompt, parse_mas_it)

    record = {
        "persona": asdict(persona),
        "model": args.model,
        "temperature": args.temperature,
        "reasoning_effort": args.reasoning_effort,
        "system_prompt": SYSTEM_PROMPT.format(persona=persona.description()),
        "association_prompt": assoc_prompt,
        "mas_it_prompt": mas_prompt,
        "associations": assoc,
        "mas_it": mas,
        "association_attempts": assoc_attempts,
        "mas_it_attempts": mas_attempts,
        "valid": assoc is not None and mas is not None,
        "seconds": round(time.time() - start, 1),
    }
    path = raw_dir / f"{persona.persona_id}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)  # never leave a half-written file behind for the resume check
    return record


def generate(args):
    from openai import OpenAI  # only needed here, so `personas` and `collect` run without it

    out_dir = Path(args.out)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    personas = read_personas(out_dir)
    if args.limit:
        personas = personas[: args.limit]

    def done(p):
        path = raw_dir / f"{p.persona_id}.json"
        return path.exists() and (json.loads(path.read_text())["valid"] or not args.redo_invalid)

    todo = [p for p in personas if not done(p)]
    print(f"{len(personas) - len(todo)} personas already done, {len(todo)} to simulate")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(simulate, client, args, p, raw_dir): p for p in todo}
        for i, future in enumerate(as_completed(futures), start=1):
            persona = futures[future]
            try:
                record = future.result()
                status = "ok" if record["valid"] else "INVALID"
                print(f"[{i}/{len(todo)}] {persona.persona_id}: {status} ({record['seconds']}s)")
            except Exception as e:
                print(f"[{i}/{len(todo)}] {persona.persona_id}: crashed: {e}")

    collect(out_dir)


# ================================================================================================
# 4. collection into the formatted-CSV schema used for the humans
# ================================================================================================


def collect(out_dir):
    out_dir = Path(out_dir)
    assoc_rows, mas_rows, report = [], [], []
    for path in sorted((out_dir / "raw").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        persona = record["persona"]
        pid = persona["persona_id"]
        report.append(
            {
                "persona_id": pid,
                "human_id": persona["human_id"],
                "sample": persona["sample"],
                "cue_variant": persona["cue_variant"],
                "is_spare": persona["is_spare"],
                "valid": record["valid"],
                "association_attempts": len(record["association_attempts"]),
                "mas_it_attempts": len(record["mas_it_attempts"]),
                "last_errors": "; ".join(
                    record["association_attempts"][-1]["errors"]
                    + record["mas_it_attempts"][-1]["errors"]
                ),
            }
        )
        if not record["valid"]:
            continue
        assoc_rows += [{"participant_id": pid, **row} for row in record["associations"]]
        mas_rows.append(
            {
                "participant_id": pid,
                **{f"mas_it_{n}": s for (n, _), s in zip(MAS_IT_ITEMS, record["mas_it"])},
            }
        )

    formatted_dir = FORMATTING / out_dir.name
    formatted_dir.mkdir(parents=True, exist_ok=True)
    write_csv(formatted_dir / "associations_valence_gpt_oss.csv", assoc_rows)
    write_csv(formatted_dir / "mas_it_gpt_oss.csv", mas_rows)
    write_csv(out_dir / "generation_report.csv", report)

    n_valid = sum(r["valid"] for r in report)
    print(f"\n{n_valid}/{len(report)} valid personas written to {formatted_dir}")
    for r in report:
        if not r["valid"]:
            print(f"  invalid: {r['persona_id']}: {r['last_errors'][:120]}")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ================================================================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("personas", help="build personas.csv from the human data")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--oversample", type=float, default=OVERSAMPLE, help="spare share per stratum")

    g = sub.add_parser("generate", help="simulate every persona in <out>/personas.csv")
    g.add_argument("--out", required=True, help="folder holding personas.csv")
    g.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"))
    g.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "not-needed"))
    g.add_argument("--model", default=os.getenv("MODEL_NAME", "openai/gpt-oss-20b"))
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None)
    g.add_argument("--max-tokens", type=int, default=8192, help="includes reasoning tokens")
    g.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    g.add_argument("--timeout", type=float, default=600)
    g.add_argument("--workers", type=int, default=1, help="parallel requests")
    g.add_argument("--limit", type=int, default=None, help="only the first N personas (tests)")
    g.add_argument("--redo-invalid", action="store_true", help="re-simulate invalid personas")

    c = sub.add_parser("collect", help="rebuild the CSVs from <out>/raw")
    c.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.command == "personas":
        build_personas(Path(args.out), args.seed, args.oversample)
    elif args.command == "generate":
        generate(args)
    else:
        collect(args.out)


if __name__ == "__main__":
    main()
