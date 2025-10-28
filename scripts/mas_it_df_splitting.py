import pandas as pd

def big_5(data):
    big_5 = data[[f'big_5_{i}' for i in range(1, 11)]].copy()

    reversed_items = [1, 4, 7, 3, 5]

    for item in reversed_items:
        big_5[f"big_5_{item}"] = 6 - big_5[f"big_5_{item}"]

    # BFI‐10 scales factors:
    extraversion = [1, 5] 
    agreeableness = [2, 7] 
    conscientiousness = [3, 8]
    neuroticism =  [4, 9]
    openness = [5, 10]

    big_5_factors = [extraversion, agreeableness, conscientiousness, neuroticism, openness]

    factors_big_5 = factor_grouping(big_5_factors, big_5, 'big_5')
    return factors_big_5

def state_anxiety(data, start, end):
    state_anxiety = data[[f'state_anxiety_{i}' for i in range(start, end)]].copy()
    return state_anxiety

def need_for_cognition(questionnaires):    
# copy the part of the dictionary that includes the scores of this questionnaire
    need_cognition = questionnaires[[f'need_cognition_{i}' for i in range(1, 19)]].copy()

    # reversed scores based on the english version: item 3, 4, 5, 7, 12, 16, 17 (which correspond to the italian version)
    reversed_items = [3, 4, 5, 7, 12, 16, 17]

    # reverse the scores
    for i in reversed_items:
        need_cognition[f'need_cognition_{i}'] = 6 - need_cognition[f'need_cognition_{i}']

    # mean
    need_cognition_mean = need_cognition.mean(axis=1)

    # descriptive analysis
    return need_cognition.describe()



def factor_grouping(factors, questionnaire, name):
        return pd.DataFrame({
        f'{name}_factor_{i + 1}': questionnaire[[f'{name}_{j}' for j in factor]].sum(axis=1)
        for i, factor in enumerate(factors)
    })

def mas_it_factor_grouping(data, sample, path):

    q_evaluation = [1,2,3,4,5,6,7,8,9]	
    q_social_anxiety = [10,11,12,13,14,15,16,17]	
    q_passive_observation = [18,19,20,21,22,23]
    total_math_axiety = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

    mas_factors = [q_evaluation, q_social_anxiety, q_passive_observation,total_math_axiety]

    mas_it_factors = factor_grouping(mas_factors, data, 'mas_it')

    mas_it_factors.insert(loc = 0, column = 'participant_id', value = data['participant_id'])
 
    mas_it_factors = mas_it_factors.rename(columns= { 'mas_it_factor_1' : 'mas_eval', 
                        'mas_it_factor_2' : 'mas_social',
                        'mas_it_factor_3' : 'mas_pass_obs',
                        'mas_it_factor_4' : 'mas_total_ma'})

    mas_it_factors.to_csv(f'{path}/out/mas_it_factors/mas_it_factors_{sample}.csv', index=False)

    describe = mas_it_factors.describe()
    skew_values = mas_it_factors.drop(columns=['participant_id']).skew()
    describe.loc['skew'] = skew_values

    descriptive_results = describe.round(2).T.drop(columns=['50%', 'min', 'max'])
    # Change row names for LaTeX table
    row_names = {
        'mas_eval': 'Evaluation MA',
        'mas_social': 'Everyday/Social MA',
        'mas_pass_obs': 'Passive Observation MA',
        'mas_total_ma': 'Total MA'
    }
    descriptive_results.index = descriptive_results.index.map(row_names)
    latex_table = descriptive_results.to_latex(escape=False, float_format="%.2f")
    print(latex_table)

def low_high_anxiety_division(mas_it_reordered, association_valence_data, sample, path):
    sum_participant = mas_it_reordered.drop(columns=['participant_id']).sum(axis=1)
    print(f"Results for {sample} sample \n")
    print(f"max total anxiety: {max(sum_participant)}")

    median_total = sum_participant.median()
    print(f"total median of the summed scores: {median_total}")

    high_anxiety = sum_participant > median_total
    print(f"number of students with high anxiety: {sum(high_anxiety)}")
    low_anxiety = sum_participant < median_total
    print(f"number of students with low anxiety: {sum(low_anxiety)}")
    to_exclude = sum_participant == median_total
    print(f"number of students to exclude: {sum(to_exclude)}")

    with open(f'{path}/out/formatting/participants_rassereno.txt', 'a') as file:
        file.write("\n"
                   f"{sample} high anxiety: {sum(high_anxiety)}\n"
                   f"{sample} low anxiety: {sum(low_anxiety)}\n")

    participant_id_high = mas_it_reordered.loc[high_anxiety, 'participant_id']
    participant_id_low = mas_it_reordered.loc[low_anxiety, 'participant_id']

    high_anxiety_data = association_valence_data[association_valence_data['participant_id'].isin(participant_id_high)]
    low_anxiety_data = association_valence_data[association_valence_data['participant_id'].isin(participant_id_low)]

    high_anxiety_data.to_csv(f'{path}/out/splitted_dfs/high_anxiety_{sample}.csv', index=False)
    low_anxiety_data.to_csv(f'{path}/out/splitted_dfs/low_anxiety_{sample}.csv', index=False)