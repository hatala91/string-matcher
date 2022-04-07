import collections
import re
import string
import unicodedata

from typing import Callable

import torch

_PRINTABLE_INDICES = {c: i for i, c in enumerate(string.printable)}
NUMBER_EMBEDDINGS = len(_PRINTABLE_INDICES) + 1
LETTER_INDEX_MAP = collections.defaultdict(
    lambda: NUMBER_EMBEDDINGS - 1, _PRINTABLE_INDICES
)


def unicode_to_ascii(s: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFD", s)
        if unicodedata.category(char) != "Mn" and char in string.printable
    )

def tokenize(word: set) -> torch.Tensor:
    return torch.tensor(
        [LETTER_INDEX_MAP[unicode_to_ascii(char)] for char in word],
        dtype=torch.long
    )


def clean(string: str) -> str:
    string = unicode_to_ascii(string).lower()
    
    string = (
        string
        .replace("&", "and")
        .replace(",", " ")
        .replace("-", " ")
    )
    string = re.sub(' +', ' ', string).strip()
    
    string = " " + string + " "
    
    string = re.sub(r'[,-./]|\sBD',r'', string)
    return string
