"""Persona manifest for the humans who got no persona in the main GPT-OSS run.

gpt_oss_data_generation.py built and simulated its personas before two changes to the human sample:
- the fisica_lecce students turned out to be high school students from southern Italy, interested
  in studying physics: they are now high_school_south and need personas with a high school profile;
- the 10 fisica_unitn students (2026-05-04 session) joined the physics sample. Seven of them are
  paired with a spare persona of the same stratum and gender (see merge_and_split_datasets.ipynb);
  the spares do not cover the other three (all male), who get a new persona here.

The new personas follow the rules of the main run (same cue list as their human, profile copied from
the human where the data has it, the rest at random). They are added to the main manifest,
rassereno/data/gpt_oss_2026/personas.csv (replacing earlier copies of the same personas), so that
`generate` simulates only them and `collect` writes every persona to the same CSVs:

    python src/gpt_oss_extra_personas.py
    python src/gpt_oss_data_generation.py generate --out rassereno/data/gpt_oss_2026 \
        --base-url http://127.0.0.1:8000/v1 --model openai/gpt-oss-20b --workers 16

Running `gpt_oss_data_generation.py personas` again would rewrite the manifest without these personas:
run this script afterwards.
"""

import random
from dataclasses import asdict

import gpt_oss_data_generation as gen
from config import RESSERENO_ROOT

OUT_DIR = RESSERENO_ROOT / "data" / "gpt_oss_2026"

# groups the main run did not know about
gen.HUMAN_GROUPS["fisica_unitn"] = ("physics", "al corso di laurea in fisica")
gen.HUMAN_GROUPS["high_school_south"] = ("high_school", "alla scuola superiore")
gen.EXCLUDED_PREFIXES = tuple(p for p in gen.EXCLUDED_PREFIXES if p != "fisica_unitn")

rng = random.Random(gen.SEED)
humans = gen.load_humans()
new_humans = humans[
    humans["participant_id"].str.startswith("high_school_south")
    | humans["participant_id"].isin(["fisica_unitn_008", "fisica_unitn_009", "fisica_unitn_011"])
].sort_values("participant_id")
personas = [
    gen.make_persona(h, f"gpt_oss_{h.participant_id}", False, rng)
    for h in new_humans.itertuples()
]

# keep the main run's personas, replace any earlier copy of these ones
new_ids = {p.persona_id for p in personas}
manifest = [r for r in gen.read_personas(OUT_DIR) if r.persona_id not in new_ids]
gen.write_csv(OUT_DIR / "personas.csv", [asdict(p) for p in manifest + personas])
print(f"{len(personas)} personas added to {OUT_DIR / 'personas.csv'} ({len(manifest)} kept)")
for p in personas:
    print(p.persona_id, p.sample, p.cue_variant, p.gender, p.age, p.programme)
