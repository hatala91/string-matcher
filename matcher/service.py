import pandas as pd


def parse_entities(df: pd.DataFrame) -> pd.DataFrame:
    current = None
    corpora = []
    for word in df["Line"].tolist():
        if word.startswith("$"):
            current = word
        else:
            corpora.append((current[1:], word))

    return pd.DataFrame(corpora, columns=["word", "misspell"])


def sample_positives(corpora_df: pd.DataFrame) -> pd.DataFrame:
    positive_df = corpora_df.copy()
    word_len = corpora_df["word"].str.len()
    misspell_len = corpora_df["misspell"].str.len()
    positive_df = positive_df[abs(word_len - misspell_len) <= 2]
    positive_df["y"] = 1
    return positive_df


def is_similar(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 2:
        return False
    if bool(set(a) - set(b) and set(b) - set(a)):
        # There is at least one character in a that is not in b ...
        # and one character in b that is not in a
        return False
    
    return True


def get_negatives(corpora_df: pd.DataFrame) -> pd.DataFrame:
    negative_df = corpora_df.copy()
    negative_df["misspell"] = np.random.permutation(negative_df["misspell"].values)
    similar = negative_df.apply(lambda row: is_similar(row[0], row[1]), axis=1)
    negative_df = negative_df[~similar.values]
    negative_df["y"] = 0
    return negative_df


def train_test_split(
    positive_df: pd.DataFrame,
    negative_df: pd.DataFrame
    train_size: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_words_df = pd.concat([positive_df, negative_df])
    unique_words = all_words_df["word"].unique()
    train_words = np.random.choice(
        unique_words,
        size=int(unique_words.size * train_size),
        replace=False
    )

    train_df = all_words_df[all_words_df["word"].isin(train_words)]
    test_df = all_words_df[~all_words_df["word"].isin(train_words)]

    return train_df, test_df
