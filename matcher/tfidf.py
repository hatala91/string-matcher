from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

from .tokenizer import clean

def get_tfidf_score(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 3),
        preprocessor=clean
    )
    vectorizer.fit(train_df["word"])

    return roc_auc_score(test_df["y"], y_prob)
