from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import pandas as pd

def jaccard_similarity(set1, set2):
    # intersection of two sets
    intersection = len(set1.intersection(set2))
    # Unions of two sets
    union = len(set1.union(set2))
    
    return intersection / union

def get_sets_of_associations(fmnt):
    associations_anxiety = set()
    associations_math = set()
    associations_exam = set()

    # for idx in range(len(fmnt)):
    for edges_list in fmnt:
        for edge in edges_list:
            cue_word, association = edge
            if cue_word == 'ansia':
                associations_anxiety.add(association)
            elif cue_word == 'matematica':
                associations_math.add(association)
            elif cue_word == 'esame':
                associations_exam.add(association)
                
    return associations_anxiety, associations_math, associations_exam

def results_jaccard_similarities(fmnts, title_group1, title_group2):   
    group1_anxiety_words, group1_math_words, group1_exam_words = get_sets_of_associations(fmnts[title_group1])
    group2_anxiety_words, group2_math_words, group2_exam_words = get_sets_of_associations(fmnts[title_group2])

    math_anxiety_similarity_group1 = jaccard_similarity(group1_math_words, group1_anxiety_words)
    math_exam_similarity_group1 = jaccard_similarity(group1_math_words, group1_exam_words)
    exam_anxiety_similiraty_group1 = jaccard_similarity(group1_exam_words, group1_anxiety_words)
    print(
        f"{title_group1}:\n"
        f"matematica - ansia: {math_anxiety_similarity_group1}\n"
        f"matematica - esame: {math_exam_similarity_group1}\n"
        f"esame - ansia: {exam_anxiety_similiraty_group1}\n"
    )
    math_anxiety_similarity_group2 = jaccard_similarity(group2_math_words, group2_anxiety_words)
    math_exam_similarity_group2 = jaccard_similarity(group2_math_words, group2_exam_words)
    exam_anxiety_similiraty_group2 = jaccard_similarity(group2_exam_words, group2_anxiety_words)
    print(
        f"{title_group2}:\n"
        f"matematica - ansia: {math_anxiety_similarity_group2}\n"
        f"matematica - esame: {math_exam_similarity_group2}\n"
        f"esame - ansia: {exam_anxiety_similiraty_group2}"
    )

    return {title_group1: (math_anxiety_similarity_group1, math_exam_similarity_group1, exam_anxiety_similiraty_group1), 
            title_group2: (math_anxiety_similarity_group2, math_exam_similarity_group2, exam_anxiety_similiraty_group2)}

def plot_comparison(df, comp, title, col1, col2, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    width = 0.35
    x = range(len(df[df['source'] == 'GPT-oss']))

    colors = {
        'GPT-oss': col1,  
        'Humans': col2 
    }
    
    for j, source in enumerate(['GPT-oss', 'Humans']):
        subset = df[df['source'] == source]
        plt.bar(
            [p + j * width for p in x],
            subset[comp],
            width=width,
            label=source,
            color=colors[source]
        )

    plt.xticks([p + width/2 for p in x], df[df['source'] == 'GPT-oss']['group'], rotation=45, ha='right')
    plt.ylabel('Jaccard Index')
    plt.title(title)
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.show()

    fig.savefig(f'{path}/out/analysis/comparison_plots/{comp}.png')