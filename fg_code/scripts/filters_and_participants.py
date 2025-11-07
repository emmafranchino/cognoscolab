import pandas as pd
import os
from pathlib import Path

high_anxiety_participant_prefixes = ['third_year_032', 'first_year_026', 'student_exp1_006', 'student_exp1_019', 'second_year_006', 'third_year_046', 'first_year_006', 'first_year_009', 'second_year_056', 'student_exp2_015', 'student_exp2_027', 'first_year_007', 'first_year_027', 'student_exp2_052', 'second_year_011', 'student_exp2_079', 'third_year_041', 'student_exp2_059', 'student_exp1_030', 'second_year_050', 'second_year_003', 'first_year_028', 'student_exp2_070', 'third_year_042', 'third_year_013', 'student_exp2_002', 'first_year_032', 'student_exp1_061', 'first_year_037', 'third_year_034', 'first_year_014', 'interfacce_010', 'third_year_031', 'third_year_019', 'third_year_038', 'student_exp1_009', 'student_exp2_065', 'first_year_013', 'third_year_030', 'third_year_014', 'student_exp1_048', 'third_year_052', 'second_year_008', 'first_year_003', 'student_exp1_038', 'student_exp2_005', 'first_year_031', 'first_year_046', 'first_year_011', 'first_year_021', 'student_exp1_063', 'interfacce_013', 'interfacce_020', 'student_exp1_032', 'student_exp1_065', 'student_exp2_019', 'third_year_051', 'student_exp1_020', 'interfacce_016', 'second_year_066', 'first_year_030', 'second_year_013', 'second_year_004', 'student_exp1_049', 'interfacce_001', 'interfacce_007', 'student_exp2_049', 'fisica_015', 'student_exp1_068', 'third_year_035', 'fisica_stem_002', 'student_exp2_061', 'student_exp2_038', 'student_exp2_023', 'student_exp2_025', 'first_year_008', 'student_exp1_031', 'student_exp2_033', 'second_year_070', 'second_year_065', 'student_exp2_008', 'student_exp2_051', 'second_year_049', 'student_exp1_011', 'student_exp1_044', 'student_exp1_062', 'student_exp1_034', 'student_exp1_017', 'student_exp2_043', 'second_year_026', 'student_exp2_071', 'first_year_029', 'third_year_028', 'student_exp2_075', 'first_year_025', 'student_exp2_017', 'student_exp1_025', 'student_exp1_039', 'second_year_053', 'student_exp2_007', 'student_exp2_006', 'second_year_012', 'student_exp1_072', 'interfacce_014', 'first_year_012', 'student_exp1_041', 'second_year_016', 'student_exp2_020', 'third_year_008', 'student_exp1_051', 'student_exp1_067', 'interfacce_019', 'third_year_033', 'student_exp2_018', 'second_year_069', 'second_year_064', 'student_exp1_053', 'third_year_021', 'student_exp1_026', 'second_year_022', 'second_year_005', 'first_year_002', 'student_exp1_007', 'third_year_004', 'second_year_018', 'student_exp1_003', 'second_year_031', 'student_exp1_024', 'first_year_044', 'student_exp1_045', 'fisica_003', 'fisica_stem_003', 'third_year_011', 'second_year_009', 'second_year_059', 'interfacce_009', 'second_year_068', 'student_exp1_054', 'student_exp2_029', 'third_year_015', 'first_year_049', 'second_year_029', 'second_year_025', 'student_exp2_036', 'student_exp2_060', 'second_year_047', 'first_year_048', 'student_exp1_029', 'student_exp1_059', 'student_exp2_013', 'student_exp2_042', 'second_year_038', 'first_year_017', 'first_year_016']
low_anxiety_participant_prefixes = ['student_exp1_002', 'student_exp1_023', 'first_year_034', 'student_exp1_074', 'student_exp1_018', 'student_exp2_074', 'student_exp1_066', 'second_year_048', 'second_year_021', 'third_year_020', 'second_year_035', 'second_year_067', 'student_exp1_046', 'second_year_036', 'third_year_027', 'first_year_042', 'third_year_029', 'third_year_040', 'first_year_010', 'third_year_026', 'fisica_007', 'student_exp2_012', 'student_exp2_076', 'third_year_047', 'fisica_016', 'third_year_012', 'second_year_058', 'fisica_012', 'third_year_009', 'student_exp2_078', 'student_exp1_012', 'first_year_043', 'student_exp1_069', 'interfacce_006', 'third_year_017', 'first_year_001', 'first_year_019', 'student_exp2_014', 'interfacce_011', 'second_year_015', 'second_year_051', 'student_exp1_050', 'second_year_045', 'fisica_stem_001', 'student_exp2_026', 'student_exp1_015', 'second_year_063', 'third_year_024', 'second_year_001', 'third_year_005', 'third_year_045', 'second_year_007', 'third_year_001', 'student_exp2_041', 'student_exp2_073', 'second_year_014', 'student_exp1_005', 'student_exp1_064', 'first_year_047', 'second_year_010', 'interfacce_012', 'student_exp1_016', 'first_year_040', 'interfacce_002', 'student_exp1_075', 'third_year_050', 'second_year_057', 'third_year_002', 'student_exp2_066', 'second_year_032', 'second_year_055', 'first_year_036', 'second_year_043', 'third_year_039', 'first_year_024', 'second_year_054', 'student_exp2_047', 'student_exp1_036', 'student_exp1_073', 'student_exp1_056', 'student_exp2_016', 'student_exp1_057', 'second_year_030', 'interfacce_004', 'student_exp1_058', 'student_exp1_060', 'interfacce_021', 'student_exp1_033', 'third_year_007', 'second_year_061', 'first_year_005', 'second_year_002', 'student_exp1_001', 'student_exp2_011', 'student_exp1_010', 'third_year_043', 'first_year_018', 'student_exp1_014', 'third_year_044', 'second_year_020', 'interfacce_022', 'student_exp1_022', 'student_exp2_022', 'second_year_024', 'second_year_037', 'second_year_017', 'student_exp2_030', 'first_year_022', 'student_exp1_043', 'second_year_044', 'third_year_049', 'interfacce_015', 'student_exp2_064', 'interfacce_008', 'student_exp1_035', 'student_exp2_035', 'student_exp2_077', 'second_year_052', 'third_year_023', 'third_year_003', 'student_exp2_063', 'third_year_022', 'first_year_038', 'interfacce_005', 'first_year_041', 'student_exp1_055', 'interfacce_017', 'first_year_033', 'second_year_060', 'interfacce_003', 'student_exp2_028', 'student_exp2_068', 'first_year_039', 'second_year_028', 'fisica_005', 'second_year_042', 'third_year_016', 'second_year_040', 'student_exp2_024', 'student_exp2_021', 'third_year_018', 'third_year_025', 'student_exp1_028', 'third_year_006', 'second_year_046', 'first_year_020', 'student_exp1_008', 'student_exp2_010', 'second_year_041', 'third_year_036', 'second_year_034', 'fisica_010', 'student_exp1_071', 'interfacce_018', 'first_year_023', 'third_year_037', 'student_exp1_047']

# Create a function to filter participant IDs based on startswith(labels).
# Labels is a list of prefixes to filter participant IDs.
# ALso filter based on if participant ID is in high or low anxiety lists.
def filter_participant_ids(participant_prefixes, labels, anxiety_level='any'):
    filtered_ids = []
    for prefix in participant_prefixes:
        if any(prefix.startswith(label) for label in labels):
            if anxiety_level == 'high' and prefix in high_anxiety_participant_prefixes:
                filtered_ids.append(prefix)
            elif anxiety_level == 'low' and prefix in low_anxiety_participant_prefixes:
                filtered_ids.append(prefix)
            elif anxiety_level == 'any':
                filtered_ids.append(prefix)
    return filtered_ids


# Create a function that extracts all column's minimum common label prefixes (the shortest non-unique common starting substrings) from a pandas DataFrame.
def get_common_label_prefixes(df_path, column_name):
    df = pd.read_csv(df_path)
    participant_ids = df[column_name].unique().tolist()

    common_prefixes = set()
    for pid in participant_ids:
        for i in range(1, len(pid) + 1):
            prefix = pid[:i]
            if all(other_pid.startswith(prefix) for other_pid in participant_ids if other_pid != pid):
                common_prefixes.add(prefix)
                break
            print(common_prefixes)
    
    # Remove the dataframe to free memory
    del df


# a function that lists all unique participant ID labels from a pandas DataFrame column.
def list_unique_participant_ids(df_path, column_name):
    if not os.path.exists(df_path):
        print(f"[WARN] Missing file: {df_path}")
        return []
    try:
        df = pd.read_csv(df_path, usecols=[column_name])
    except ValueError:
        # Column missing: load minimally and try to find a close match
        df = pd.read_csv(df_path, nrows=5)
        raise KeyError(f"'participant_id' column not found in {df_path}. Columns: {list(df.columns)}")
    participant_ids = df[column_name].astype("string").dropna().unique().tolist()
    del df
    return participant_ids

# Create filtered lists dictionary
# Build from the 12 files visible in the provided screenshot

# Build base path robustly from this script location:
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root (parent of scripts/)
base_dir = PROJECT_ROOT / 'out' / 'formatting' / 'splitted_dfs'

dataset_files = {
    # Psychology (ALL)
    'psychology_high_anxiety':        'high_anxiety_all_psychology_students.csv',
    'psychology_low_anxiety':         'low_anxiety_all_psychology_students.csv',

    # Psychology (gpt_oss)
    'psychology_high_anxiety_gpt_oss': 'high_anxiety_psychology_students_gpt_oss.csv',
    'psychology_low_anxiety_gpt_oss':  'low_anxiety_psychology_students_gpt_oss.csv',

    # Physics (2025)
    'physics_high_anxiety_2025':      'high_anxiety_physics_students_2025.csv',
    'physics_low_anxiety_2025':       'low_anxiety_physics_students_2025.csv',

    # Physics (gpt_oss)
    'physics_high_anxiety_gpt_oss':    'high_anxiety_physics_students_gpt_oss.csv',
    'physics_low_anxiety_gpt_oss':     'low_anxiety_physics_students_gpt_4o.csv',

    # High school (2025)
    'highschool_north_high_anxiety_2025': 'high_anxiety_high_school_students_2025.csv',
    'highschool_north_low_anxiety_2025':  'low_anxiety_high_school_students_2025.csv',

    # High school (gpt_oss)
    'highschool_high_anxiety_gpt_oss': 'high_anxiety_high_school_students_gpt_oss.csv',
    'highschool_low_anxiety_gpt_oss':  'low_anxiety_high_school_students_gpt_oss.csv',
}

filters = {}
for key, fname in dataset_files.items():
    path = str(base_dir / fname)
    filters[key] = list_unique_participant_ids(path, 'participant_id')

# We do not have the participant_ids for the following datasets,
# so we will just print their counts as provided.
print("highschool_south_2019: 159")
print("researchers_2019: 59")
# others

print("------------")
print("PARTICIPANTS")
print("------------")
for key, value in filters.items():
    print(f'{key}: {len(value)}')

# Grand total across all datasets (unique over every file)
all_unique_participants = sorted(set(pid for lst in filters.values() for pid in lst))
print(f"\nTotal unique participants across all listed datasets: {len(all_unique_participants)}\n")

print("Detailed participant IDs in each filter:\n")
for key, labels in filters.items():
    print(f'Filter: {key}, Number of participant IDs: {len(labels)}')
    print(f"{labels}\n")