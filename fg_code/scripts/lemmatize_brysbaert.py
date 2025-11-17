import os
import sys

# Add project root (parent of this scripts/ folder) to sys.path once
PROJECT_ROOT = os.path.abspath("..")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Now import from ../src folder
from src.models_and_vars import load_env_vars, ensure_nltk_path, init_spacy_and_emo
from src.data_processing import lemmatize_brysbaert

# you do not need to run these download commands every time. nltk.download()
# saves these data packages to your local file system in the nltk_data directory.
# Once they are downloaded, they are available for NLTK to use in any session.

'''nltk.download('wordnet')
nltk.download('omw-1.4') # for italian and multilingual support, if needed
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng') # for word tokenization
nltk.download('averaged_perceptron_tagger') # for POS tagging
'''

# Set language once; everything else adapts
language = "italian"  # or "english"
env_file_path = "../env"

# Execute setup
load_env_vars(env_file_path)
ensure_nltk_path()

nlp, emo = init_spacy_and_emo(language)

# Also preload both models for explicit use
nlp_en, _ = init_spacy_and_emo("english")
nlp_it, _ = init_spacy_and_emo("italian")

# Create english lemmatized Brysbaert dataset
lem_en = lemmatize_brysbaert(
    input_path="../data/brysbaert_concreteness_dataset_eng_ita/brysbaert_concreteness_dataset_eng_ita.csv",
    output_path="../data/brysbaert_concreteness_dataset_eng_ita/brysbaert_concreteness_dataset_eng_ita_lemmatized_en.csv",
    word_column="english",
    language="en",
    overwrite=True,
    nlp=nlp_en
)

# Create italian lemmatized Brysbaert dataset
lem_it = lemmatize_brysbaert(
    input_path="../data/brysbaert_concreteness_dataset_eng_ita/brysbaert_concreteness_dataset_eng_ita.csv",
    output_path="../data/brysbaert_concreteness_dataset_eng_ita/brysbaert_concreteness_dataset_eng_ita_lemmatized_it.csv",
    word_column="italian",
    language="it",
    overwrite=True,
    nlp=nlp_it
)

print("Lemmatized Brysbaert datasets created.")