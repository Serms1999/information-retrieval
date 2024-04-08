from typing import List, Tuple, Union, Set
from pandas import Series
from nltk.tokenize import RegexpTokenizer


def lowercase_data(data: Union[Series, List[str], str]) -> Union[Series, List[str], str]:
    """
    Lowercase the data.

    Parameters
    ----------
    data : Union[Series, List[str], str]
        The data to lowercase.

    Returns
    -------
    Union[Series, List[str], str]
        The data with all strings lowercased.
    """
    if isinstance(data, Series):
        return data.str.lower()
    elif isinstance(data, List):
        return [x.lower() for x in data]
    elif isinstance(data, str):
        return data.lower()
    else:
        raise ValueError('Data type not supported.')


def remove_punctuation(data: Union[Series, List[str], str]) -> Union[Series, List[str], str]:
    """
    Remove punctuation from the data.

    Parameters
    ----------
    data : Union[Series, List[str], str]
        The data to remove punctuation from.

    Returns
    -------
    Union[Series, List[str], str]
        The data with all punctuation removed.
    """
    tokenizer = RegexpTokenizer(r'[a-z]+')
    if isinstance(data, Series):
        return data.apply(lambda x: ' '.join(tokenizer.tokenize(x)))
    elif isinstance(data, List):
        return [' '.join(tokenizer.tokenize(x)) for x in data]
    elif isinstance(data, str):
        return ' '.join(tokenizer.tokenize(data))
    else:
        raise ValueError('Data type not supported.')


def remove_stopwords(data: Union[Series, List[str], str], stopwords: Set[str]) -> Union[Series, List[str], str]:
    """
    Remove stopwords from the data.

    Parameters
    ----------
    data : Union[Series, List[str], str]
        The data to remove stopwords from.
    stopwords : Set[str]
        The set of stopwords to remove.

    Returns
    -------
    Union[Series, List[str], str]
        The data with all stopwords removed.
    """
    if isinstance(data, Series):
        return data.apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
    elif isinstance(data, List):
        return [' '.join([word for word in x.split() if word not in stopwords]) for x in data]
    elif isinstance(data, str):
        return ' '.join([word for word in data.split() if word not in stopwords])
    else:
        raise ValueError('Data type not supported.')
