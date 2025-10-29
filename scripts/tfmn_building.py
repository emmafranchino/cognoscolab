import re
from emoatlas import EmoScores
import matplotlib.pyplot as plt

emo = EmoScores(language="italian", spacy_model='it_core_news_lg')

def text_cleaning(df):
    # joins all the string by adding '.'
    joined_text = ".".join(df['text'])
    # removes the unnecessary sentence of the instruction (case insensitively)
    cleaned_text = re.sub(r'(?i)quando affronto le discipline STEM', '', joined_text)
    # returns cleaned text
    return cleaned_text

def tfmn_buiding(texts, keywords, folder_path, sample, flower = 'yes'):    
    for keyword in keywords:
        try: 
            # builds the forma mentis network assigning them a title and inserting them into a dict
            fig, ax = plt.subplots(figsize=(14,14))
            ax.axis('off')
            fmnt = emo.formamentis_network(texts, target_word=keyword)
            # draws the fmnt and saves it in a folder inserting in the file name the number of vertices of the fmnt
            emo.draw_formamentis(fmnt,highlight=keyword,ax=ax)
            fig.savefig(f'{folder_path}/out/analysis/tfmn/{sample}/{keyword}', bbox_inches="tight", pad_inches=0.1)
            
            if flower == 'yes':
                zeta_scores = emo.zscores(fmnt)
                emo.draw_plutchik(zeta_scores, reject_range=(-1.96, 1.96))
                plt.savefig(f"{folder_path}/out/analysis/emo_flowers_tfmn/{sample}/{keyword}", bbox_inches="tight", pad_inches=0.1)
                plt.close('all')

        # if the data is not sufficient a ValueError will rise, so we skip that fmnt and move to the next one
        except ValueError as e:
            if "min() iterable argument is empty" in str(e):
                print(f"⚠️ Skipping '{keyword}' of {sample} sample: not enough data to build the network.")
            else:
                raise