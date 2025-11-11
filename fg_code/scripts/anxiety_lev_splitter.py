from pathlib import Path
import pandas as pd

# Resolve project root and data dir regardless of current working directory
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'


def low_high_anxiety_division(mas_it_reordered, association_valence_data, sample, path):
    sum_participant = mas_it_reordered.drop(columns=['participant_id']).sum(axis=1)
    print("========================================")
    print(f"{sum_participant}\n")
    print(f"Results for {sample} sample \n")
    print(f"max total anxiety: {max(sum_participant)}")

    median_total = sum_participant.median()
    print(f"total median of the summed scores: {median_total}")

    # determine high and low anxiety based on median
    high_anxiety = sum_participant > median_total
    print(f"number of students with high anxiety: {sum(high_anxiety)}")
    low_anxiety = sum_participant < median_total
    print(f"number of students with low anxiety: {sum(low_anxiety)}")
    to_exclude = sum_participant == median_total
    print(f"number of students to exclude: {sum(to_exclude)}")

    # create directory for splitted dfs if it doesn't exist
    Path(f'{path}/splitted_dfs/{sample}').mkdir(parents=True, exist_ok=True)

    with open(f'{path}/splitted_dfs/{sample}/splitted.txt', 'a') as file:
        file.write("\n"
                   f"{sample} high anxiety: {sum(high_anxiety)}\n"
                   f"{sample} low anxiety: {sum(low_anxiety)}\n")

    # get participant ids for high and low anxiety
    participant_id_high = mas_it_reordered.loc[high_anxiety, 'participant_id']
    participant_id_low = mas_it_reordered.loc[low_anxiety, 'participant_id']

    # filter association_valence_data based on participant ids for high and low anxiety 
    high_anxiety_data = association_valence_data[association_valence_data['participant_id'].isin(participant_id_high)]
    low_anxiety_data = association_valence_data[association_valence_data['participant_id'].isin(participant_id_low)]

    high_anxiety_data.to_csv(f'{path}/splitted_dfs/{sample}/high_anx.csv', index=False)
    low_anxiety_data.to_csv(f'{path}/splitted_dfs/{sample}/low_anx.csv', index=False)

    return high_anxiety_data, low_anxiety_data


mas_it_all_psy_students = pd.read_csv(f'{DATA_DIR}/mas_it_all_psy_students.csv')
wide_psy_associations_valences = pd.read_csv(f'{DATA_DIR}/associations_valence_all_psy_students.csv')

low_high_anxiety_division(mas_it_all_psy_students, wide_psy_associations_valences, 'all_psychology_students', DATA_DIR)