import os
import spacy
from emoatlas import EmoScores
import nltk


def load_env_vars(path: str, language: str | None = None) -> None:
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    key, value = s.split("=", 1)
                    os.environ[key] = value
        # Print the actually defined model path variables
        print(f"Loaded SPACY_MODEL_PATH_EN: {os.environ.get('SPACY_MODEL_PATH_EN')}")
        print(f"Loaded SPACY_MODEL_PATH_IT: {os.environ.get('SPACY_MODEL_PATH_IT')}")
        print(f"Loaded NLTK_DATA: {os.environ.get('NLTK_DATA')}\n")
        # (Optional) set a generic SPACY_MODEL_PATH to the selected language for backward compatibility
        if language:
            lang_var = "SPACY_MODEL_PATH_EN" if language.lower().startswith("en") else "SPACY_MODEL_PATH_IT"
            chosen = os.environ.get(lang_var)
            if chosen:
                os.environ.setdefault("SPACY_MODEL_PATH", chosen)
    else:
        print(f"Warning: {path} file not found. Using default paths.")


def ensure_nltk_path():
    nltk_data_dir = os.environ.get("NLTK_DATA")
    if nltk_data_dir:
        try:
            import nltk
            if nltk_data_dir not in nltk.data.path:
                nltk.data.path.insert(0, nltk_data_dir)
        except Exception as e:
            print(f"Warning: could not set NLTK_DATA path: {e}")


def init_spacy_and_emo(language: str):
    var = "SPACY_MODEL_PATH_EN" if language.lower().startswith("en") else "SPACY_MODEL_PATH_IT"
    model_path = os.environ.get(var)
    if not model_path:
        raise RuntimeError(f"{var} not set in env file. Please define it before running.")
    try:
        nlp = spacy.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load spaCy model at '{model_path}': {e}")
    emo = EmoScores(language=language, spacy_model=model_path)
    return nlp, emo