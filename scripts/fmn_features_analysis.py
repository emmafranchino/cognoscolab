import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm

def network_features_extraction(sample, features, mas_it_factors_path, features_bfmn_path):
    mas_it_factors = pd.read_csv(f'{mas_it_factors_path}/mas_it_factors_{sample}.csv')

    features_total= mas_it_factors.merge(features, on='participant_id', how='left')
    features_total = features_total.drop(columns=['participant_id'])
    features_total.to_csv(f'{features_bfmn_path}/features_total_{sample}.csv', index=False)

    results = features_total.describe().T.drop(columns=['25%','50%', '75%', 'min', 'max'])
    results = results.drop(index=['mas_total_ma', 'mas_social', 'mas_eval', 'mas_pass_obs', 'valence_ansia', 'valence_matematica'])
    row_names = {
            'degree_ansia': '"Anxiety" Degree',
            'clo_cen_ansia': '"Anxiety" Closeness Centrality',
            'degree_matematica': '"Math" Degree',
            'clo_cen_matematica': '"Math" Closeness Centrality'
        }
    results.index = results.index.map(row_names)
    latex_table = results.to_latex(escape=False, float_format="%.2f")

    print(latex_table)
    return features_total

def plot_correlogram(df, file_name, folder_path):
    df_copied = df.copy()
    df_copied = df_copied[['mas_total_ma', 'mas_eval', 'mas_social', 'mas_pass_obs',
                            'valence_ansia', 'degree_ansia', 'clo_cen_ansia',
                            'valence_matematica', 'degree_matematica', 'clo_cen_matematica']]
    # change col names for visualization
    df_copied.columns = [
    "Total MA",
    "Evaluation MA",
    "Social MA",
    "Observation MA",
    "Valence \"anxiety\"",
    "Degree \"anxiety\"",
    "CC \"anxiety\"",
    "Valence \"math\"",
    "Degree \"math\"",
    "CC \"math\""
    ]

    # compute pearson correlation matrix
    corr = df_copied.corr(method='pearson')

    # inizialize df for p-values
    annot = corr.copy()

    # compute p-values for each pair of variables, with their signficance stars
    for i in corr.columns:
        for j in corr.columns:
            if i != j:
                _, p = stats.pearsonr(df_copied[i], df_copied[j])

                if p < 0.001:
                    p = '***'
                elif p < 0.01:
                    p = '**'
                elif p < 0.05:
                    p = '*'
                else:
                    p = ''
                annot.loc[i, j] = f"{corr.loc[i, j]:.2f}{p}"
            else:
                annot.loc[i, j] = f"{corr.loc[i, j]:.2f}"

    # plot the correlogram
    plt.figure(figsize=(12, 10))
    # keep only the upper triangle
    matrix = np.triu(np.ones_like(corr, dtype=bool))
    # plot heatmap with corr values, p-values with their stars, and the bar for colors
    sns.heatmap(corr, annot=annot, annot_kws={"size":12},fmt='', cmap="crest", 
                mask = matrix,
                cbar_kws={"orientation": "horizontal", 'shrink': 0.6, 'location' : 'top', "pad": 0}
                )
    # rotate the x and y ticks for better visualization
    plt.xticks(rotation=45, ha='right')   
    plt.yticks(rotation=45, ha='right')    
    plt.tight_layout()
    # save the plot
    plt.savefig(f'{folder_path}/out/analysis/correlograms/{file_name}.png', bbox_inches="tight")
    plt.close()

def stand_vars_lin(data):

    scaler = MinMaxScaler()

    lin_reg_vars = data[['mas_total_ma', 'mas_social', 'mas_eval', 'mas_pass_obs', 'valence_ansia', 'degree_ansia', 'valence_matematica', 'degree_matematica']].copy()

    stand_data = pd.DataFrame()
    for col in lin_reg_vars.columns:
        scaled_col = scaler.fit_transform(lin_reg_vars[[col]])
        stand_data[col] = scaled_col.flatten()

    return stand_data

def lin_reg_ols_predictors(stand_data, factor, predictors):
    # depentent variable
    Y = stand_data[factor]
    # parameters
    X = stand_data[predictors]
    # intercept
    X = sm.add_constant(X) 

    model = sm.OLS(Y, X).fit()

    results = model.summary2().tables[1]

    # function to format p-values with asterisks
    def format_p(p):
        formatted_p = f"{p:.3f}".lstrip("0") # keeps 3 decimals and removes leading zeros
        if p < 0.001:
            return '< .001***'
        if p < 0.01:
            return f"{formatted_p}**"
        elif p < 0.05:
            return f"{formatted_p}*"
        else:
            return formatted_p

    # take only the columns of interest
    results = results[['Coef.', 'Std.Err.', 't', 'P>|t|']]
    # keep the first 3 decimals of the coefficients, standard errors and t-values
    col_to_format = ['Coef.', 'Std.Err.', 't']
    # for all the results keep only 3 decimals and remove leading zeros
    results[col_to_format] = results[col_to_format].applymap(lambda x: f"{x:.3f}".lstrip("0"))
    # apply the p-value formatting function to the 'P>|t|' column
    results['P>|t|'] = results['P>|t|'].apply(format_p)

    # rename the indices of the table based on the predictors used
    if len(predictors) == 4:
        results.index = ['Intercept', '"Anxiety" Degree', '"Math" Degree', '"Anxiety" Valence', '"Math" Valence']
    elif len(predictors) == 3:
        results.index = ['Intercept', '"Anxiety" Degree', '"Anxiety" Valence', '"Math" Valence']
        
    # build latex table to be inserted in overleaf
    latex_table = results.to_latex(escape=False)  # keep asterisks

    # print the R-squared value and latex table
    print(f"R-squared {factor} {len(predictors)} predictors: {model.rsquared:.3f}\nAIC {factor} {len(predictors)} predictors: {model.aic}\n")
 
    # if we wanto to see the latex table we can print it
    return model.aic, latex_table

def lin_regression_results(stand_data, model_name, factors):
    for factor in factors:
        print(f'## Results for {model_name} {factor} ##\n')
        aic_4_factors, table_4_factors = lin_reg_ols_predictors(stand_data, factor, ['degree_ansia', 'degree_matematica', 'valence_ansia','valence_matematica'])
        aic_3_factors, table_3_factors = lin_reg_ols_predictors(stand_data, factor, ['degree_ansia', 'valence_ansia', 'valence_matematica']) # perché negli umani il degree di math non correla con i fattori della math anxiety

        if aic_4_factors < aic_3_factors:
            print("--> 4 predictors model is better\n")
            print(table_4_factors)
        else:
            print("--> 3 predictors model is better\n")
            print(table_3_factors)