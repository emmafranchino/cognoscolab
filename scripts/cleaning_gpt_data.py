import pandas as pd
import os
import re
import itertools
import numpy as np

def get_raw_data(folder):
    file_paths = []

    for root, dirs, files in os.walk(folder):
        for name in files:
            if name.endswith(".txt") and name.startswith('student'):
                file_paths.append(os.path.join(root, name))

    all_txts = []
    for file_path in file_paths:
        txt = open(file_path).read()
        path = re.search(r'test_studenti_gpt_4o/(.*)', file_path)
        trimmed_path = path.group(1) if path else file_path
        all_txts.append({'txt' : txt, 
                         'file_path' : trimmed_path
                         })

    all_txts = pd.DataFrame(all_txts)
    print(f"Number of generated files: {len(file_paths)}")
    return all_txts

def process_txt_dataset(all_txts, dataset_name):
    data = []

    for i, txt in enumerate(all_txts['txt']):
        # create participant_id
        participant_id = f"{dataset_name}_{i+1:03d}"

        # base text
        mod_text = txt

        # --- CLEANING ---
        # case: 4NaNNaN -> remove NaN repetition
        value_nan = re.search(r'(\d)(NaN){1,3}', mod_text)
        if value_nan:
            mod_text = mod_text.replace(value_nan.group(2), '')

        # remove standalone NaN
        mod_text = re.sub(r'\bNaN\b', '', mod_text)

        # case: determinazionenan5 -> determinazione,5
        mod_text = re.sub(r'\b(\w*?)nan(\d)\b', r'\1,\2', mod_text, flags=re.IGNORECASE)

        # remove junk text before first "fisica"
        fisica_text = re.search(r'(?im)(^.*?)(fisica,\d+)', mod_text)
        if fisica_text:
            mod_text = mod_text.replace(fisica_text.group(1), '')

        # --- PROMPT EXTRACTION ---
        prompt = None
        prompt_match = re.search(r'(?s)(.*?)(\n[Ff]isica,\d+)', mod_text)
        if prompt_match:
            prompt = prompt_match.group(1)
            # drop prompt text but keep "fisica"
            mod_text = mod_text.replace(str(prompt_match.group(1)), '')

        # --- MAS-IT EXTRACTION (23 numbers comma-separated) ---
        mas_it = None
        lines = mod_text.splitlines()
        for line in lines:
            line = line.strip()
            match = re.search(r'^\s*(?:\d+,){22}\d+\s*$', line)
            if match:
                numbers = line.split(',')
                cleaned = []
                for n in numbers:
                    num = int(n)   
                    # keep 1–5, otherwise replace with NaN
                    if 1 <= num <= 5:
                        cleaned.append(str(num))
                    else:
                        cleaned.append(str(np.nan))  # nan as str to then concatenate it
                mas_it = ','.join(cleaned)

        # --- ASSOCIATIONS EXTRACTION ---
        filtered_lines = [
            line for line in lines
            if re.match(r"^([a-zA-Z0-9_\-']+,\d+,){2,}[a-zA-Z0-9_\-']+,\d+(,*)$", line)
        ]
        mod_text = '\n'.join(filtered_lines)

        # --- COLLECT EVERYTHING ---
        data.append({
            "participant_id": participant_id,
            "file_path": all_txts["file_path"].iloc[i] if "file_path" in all_txts else None,
            "prompt": prompt,
            "mas_it": mas_it,
            "associations": mod_text,
        })

    return pd.DataFrame(data)

def extract_personal_info(df):
    
    ages, genders, enrolment_years, faculties, conditions = [], [], [], [], []

    for prompt in df['prompt'].fillna(""):
        # --- AGE ---
        age_match = re.search(r'\b(\d{1,3})\s+anni\b', prompt)
        ages.append(age_match.group(1) if age_match else None)

        # --- GENDER ---
        if re.search(r'\bSei una studentessa\b', prompt):
            genders.append("F")
        elif re.search(r'\bSei uno studente\b', prompt):
            genders.append("M")
        else:
            genders.append(None)

        # --- ENROLMENT YEAR ---
        enrolment_match = re.search(r'\b(\d+)°\s+anno', prompt)
        enrolment_years.append(enrolment_match.group(1) if enrolment_match else None)

        # --- FACULTY / SCHOOL ---
        uni_match = re.search(r'universita alla facolta di\s+(\w+)', prompt, flags=re.IGNORECASE)
        liceo_match = re.search(r'scuola superiore al liceo\s+(\w+)', prompt, flags=re.IGNORECASE)
        if uni_match:
            faculties.append(f"uni_{uni_match.group(1)}")
        elif liceo_match:
            faculties.append(f"liceo_{liceo_match.group(1)}")
        else:
            faculties.append(None)

        # --- SOCIO-ECONOMIC CONDITION ---
        socio_match = re.search(r'vivi in condizioni socio-economiche\s+([\w\-]+)', prompt, flags=re.IGNORECASE)
        conditions.append(socio_match.group(1) if socio_match else None)

    # --- Add to DataFrame ---
    df = df[['participant_id', 'file_path', 'mas_it']].copy()  # Keep only participant_id to avoid modifying original df
    df["age"] = ages
    df["gender"] = genders
    df["year"] = enrolment_years
    df["faculty"] = faculties
    df["condition"] = conditions

    return df

def all_questionnaires(df):
    
    # Split MAS-IT responses
    df_mas_it = df['mas_it'].str.split(",", expand=True)
    
    # Label the columns
    df_mas_it.columns = [f"mas_it_{i+1}" for i in range(23)]
    
    df_mas_it.insert(0, 'participant_id', df['participant_id'])

    # mapped scores with italian order to the origianl order
    correspondence_order = [
        'participant_id', 'mas_it_1', 'mas_it_3', 'mas_it_6', 'mas_it_7', 'mas_it_10', 'mas_it_18', 'mas_it_19',
        'mas_it_21', 'mas_it_23', 'mas_it_2', 'mas_it_4', 'mas_it_5', 'mas_it_8', 'mas_it_11',
        'mas_it_13', 'mas_it_14', 'mas_it_22', 'mas_it_9', 'mas_it_12', 'mas_it_15',
        'mas_it_16', 'mas_it_17', 'mas_it_20'
    ]

    # reorder columns based on the original order
    mas_it_reordered = df_mas_it[correspondence_order]

    # # name columns to match the original names
    mas_it_reordered.columns = df_mas_it.columns

    id_cols=["participant_id", "age", "gender", "faculty", "year", "condition"]
    
    # Concatenate with id/personal info
    df_questionnaire = df[id_cols]

    return df_questionnaire, mas_it_reordered

def process_associations(df):

    df_assoc = df[["participant_id", "associations"]].copy()

    # Split by newline and explode
    df_assoc["associations"] = df_assoc["associations"].str.split("\n")
    df_assoc = df_assoc.explode("associations").reset_index(drop=True)

    # Remove empty lines
    df_assoc = df_assoc[df_assoc["associations"].str.strip().astype(bool)]

    # Split into separate columns
    col_names = [
        "cue_word", "valence_cue_word",
        "association_1", "valence_association_1",
        "association_2", "valence_association_2",
        "association_3", "valence_association_3"
    ]
    df_assoc[col_names] = df_assoc["associations"].str.split(",", expand=True)

    # Drop original column
    df_assoc = df_assoc.drop(columns=["associations"])

    # Lowercase certain columns
    for col in ["cue_word", "association_1", "association_2", "association_3"]:
        df_assoc[col] = df_assoc[col].str.lower()

    # Correct known typos
    df_assoc.loc[df_assoc["cue_word"] == "psicoogia", "cue_word"] = "psicologia"

    # Sort
    df_assoc = df_assoc.sort_values(by=[ "cue_word", "participant_id"]).reset_index(drop=True)

    return df_assoc

def find_missing_cue_words(df_associations, df_personal_data):
    # Get unique cue words and participants
    cue_words = df_associations['cue_word'].unique().tolist()
    participants = df_associations['participant_id'].unique().tolist()

    print(f"Number of cue words: {len(cue_words)}")
    print(f"Number of participants: {len(participants)}")

    # Build all possible participant × cue_word pairs
    all_combinations = pd.DataFrame(
        itertools.product(participants, cue_words),
        columns=['participant_id', 'cue_word']
    )

    # Compare against actual associations
    merged = all_combinations.merge(
        df_associations[['participant_id', 'cue_word']],
        how="left", indicator=True
    )

    # Keep only missing ones
    missing_pairs = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

    # Add file_path info from metadata
    missing_info = df_personal_data[['participant_id', 'file_path']]
    missing_df = missing_pairs.merge(missing_info, on='participant_id', how='left')

    # Output
    if missing_pairs.empty:
        print("No missing cue words ✅")
    else:
        print("⚠️ Missing cue words found")
        print(missing_pairs)
    
    len_cue_words = []
    # get how many cue words are in the df, to see wich one are missing
    for cue_word in cue_words:
        len_word = df_associations.loc[df_associations['cue_word'] == cue_word, 'cue_word']
        len_cue_words.append(len(len_word))
        # if there arent 356 cue words (number of participants)
        if len(len_word) != len(df_personal_data):
            print(f'{cue_word} : {len(len_word)}')