"""MutantX-S Implementation for EMBER Function Imports

This script contains method stubs to guide Noah's planned
implementation of MutantX-S, which aims to be comparable
against COUGAR.
"""


def main():
    print('Hello, world!')
    # md5_to_ngrams = convert_function_imports_to_ngrams(info_list, record_list)
    # md5_to_fvs, int_to_ngram = create_feature_vectors_from_ngrams(md5_to_ngrams)
    # reduce_dimensions_hashing_trick()
    # select_prototypes()
    # cluster_prototypes()


def convert_function_imports_to_ngrams(info_list: list, record_list: list, N: int = 4) -> dict:
    """Converts functions imported by malware samples to N-grams
    representing potential behaviours of those samples.

    Parameters
    ----------
    info_list : list
        A list containing exactly one tuple per malware sample,
        where tuples are of the form (MD5, number of imports)
    record_list : list
        A list containing a variable number of tuples per malware
        sample, where tuples are of the form: [MD5, library, function]
    N : int, optional
        The window size for the N-grams, default 4 (from paper)

    Returns
    -------
    dict
        a mapping of str to list: MD5 to list of N-grams
    """
    pass


def create_feature_vectors_from_ngrams(sample_to_ngrams: dict) -> tuple:
    """Create feature vectors for each document, where an integer
    in the vector represents the presence of a corresponding N-gram in
    that document (malware sample).

    Parameters
    ----------
    sample_to_ngrams : dict
        A dict mapping str to lists: MD5 to list of N-grams

    Returns
    -------
    (dict, dict)
        a mapping of str to list: MD5 to feature vector (list of ints)
        a mapping of int to str: numerical encoding to N-gram
    """
    pass


def reduce_dimensions_hashing_trick():
    # If you want to work forward, look at this class for help:
    # from sklearn.feature_extraction import FeatureHasher
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html
    pass


def select_prototypes():
    pass


def cluster_prototypes():
    pass


if __name__ == '__main__':
    main()
