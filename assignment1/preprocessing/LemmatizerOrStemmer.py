from enum import Enum
from typing import List, Tuple, Union, Set
from nltk.stem import WordNetLemmatizer, PorterStemmer
from pandas import Series


class WordSimplification(Enum):
    """
    Enum to specify whether to use a lemmatizer or a stemmer.
    """
    LEMMATIZER = 1
    STEMMER = 2


class LemmatizerOrStemmer:
    """
    Class to simplify words using either a lemmatizer or a stemmer.
    """
    def __init__(self, word_simplification: WordSimplification) -> None:
        self.word_simplification = word_simplification
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def simplify(self, word: Union[Series, List[str], str]) -> Union[Series, List[str], str]:
        """
        Simplify the word using either a lemmatizer or a stemmer.

        Parameters
        ----------
        word : Union[Series, List[str], str]
            The word to simplify.

        Returns
        -------
        Union[Series, List[str], str]
            The simplified word.
        """
        if self.word_simplification == WordSimplification.LEMMATIZER:
            if isinstance(word, Series):
                return word.apply(lambda x: self.lemmatizer.lemmatize(x))
            elif isinstance(word, List):
                return [self.lemmatizer.lemmatize(x) for x in word]
            elif isinstance(word, str):
                return self.lemmatizer.lemmatize(word)
            else:
                raise ValueError("Word type not supported.")
        elif self.word_simplification == WordSimplification.STEMMER:
            if isinstance(word, Series):
                return word.apply(lambda x: self.stemmer.stem(x))
            elif isinstance(word, List):
                return [self.stemmer.stem(x) for x in word]
            elif isinstance(word, str):
                return self.stemmer.stem(word)
            else:
                raise ValueError("Word type not supported.")
        else:
            raise ValueError("Word simplification type not supported.")
