"""MutantX-S Implementation for EMBER Function Imports

This script contains method stubs to guide Noah's planned
implementation of MutantX-S, which aims to be comparable
against COUGAR.
"""
from json import loads

from scipy.sparse import csr_matrix
from sklearn.feature_extraction import FeatureHasher


def main():

    info_list, record_list = open_ember_files()

    md5_to_ngrams = convert_function_imports_to_ngrams(info_list, record_list)

    md5_to_fvs, int_to_ngram = create_feature_vectors_from_ngrams(md5_to_ngrams)

    # reduce_dimensions_hashing_trick()
    # select_prototypes()
    # cluster_prototypes()


def open_ember_files() -> tuple:
    info_list = list()
    record_list = list()

    file_name = input("Select file name: ")

    with open(file_name, 'r') as f:

        for line in f:
            json_doc = loads(line)

            md5 = json_doc['md5']
            imports = json_doc['imports']
            label = json_doc['label']

            if label == 1:
                count = 0
                for library in imports:
                    for function in imports[library]:
                        count += 1
                        record_list.append((md5, library, function))

                info_list.append((md5, str(count)))

    return info_list, record_list


def convert_function_imports_to_ngrams(info_list: list, record_list: list, n: int = 4) -> dict:
    """Converts functions imported by malware samples to N-grams
    representing potential behaviours of those samples.

    Parameters
    ----------
    info_list : list
        A list containing exactly one tuple per malware sample,
        where tuples are of the form (MD5, number of imports)
    record_list : list
        A list containing a variable number of tuples per malware
        sample, where tuples are of the form: (MD5, library, function)
    n : int, optional
        The window size for the N-grams, default 4 (from paper)

    Returns
    -------
    dict
        a mapping of str to list: MD5 to list of N-grams
    """

    import_index = 0
    n_gram_mapping = dict()

    for md5, num_imports in info_list:

        n_gram_mapping[md5] = list()

        for index in range(import_index, import_index + int(num_imports) - n + 1):

            n_gram = tuple([record[2].lower() + "," + record[1].lower() for record in record_list[index:index+n]])

            n_gram_mapping[md5].append(n_gram)

        import_index += int(num_imports)

    return n_gram_mapping


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

    # Create a set of each observed N-gram
    n_grams = {n_gram for n_gram_list in sample_to_ngrams.values() for n_gram in n_gram_list}

    # Create a unique numerical encoding for each observed N-gram
    n_gram_encodings = {k: v for k, v in zip(range(len(n_grams)), n_grams)}

    # Create feature vectors to represent each sample
    encodings_reverse_dict = {v: k for k, v in n_gram_encodings.items()}
    md5_vector_mapping = dict()

    for k, v in sample_to_ngrams.items():
        md5_vector_mapping[k] = [0] * len(n_gram_encodings)

        for n_gram in v:
            md5_vector_mapping[k][encodings_reverse_dict[n_gram]] += 1

    return md5_vector_mapping, n_gram_encodings


def reduce_dimensions_hashing_trick(md5_vector_mapping: dict, int_to_ngram: dict) -> csr_matrix:
    """Reduce dimensions to a vector of a fixed-length by
    applying the hashing trick.

    Look at this class for help:
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html
    The input_type should be 'string', and we will need to join the
    n-gram tuples into a string in a consistent manner so that they
    can be hashed.

    As well, per the MutantX-S paper: "In case of a collision where
    two or more N-grams map to the same position, the sum of their
    counts is used as the value in the new vector."
    As a result, we probably want to set alternate_sign=False, so
    that we accumulate error rather than cancelling it.

    Parameters
    ----------
    md5_vector_mapping : dict
        A mapping of str to list: MD5 to feature vector (list of ints)
    int_to_ngram: dict
        A mapping of int to str: numerical encoding to N-gram

    Returns
    -------
    X : sparse matrix of shape (n_samples, n_features)
        Feature matrix from hashed n-grams. We may densify this,
        depending on the memory requirements and ability to calculate
        Euclidean distance on sparse vectors.
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
    """

    h = FeatureHasher(2**12, alternate_sign=False)

    fv_matrix = list()

    for fv in md5_vector_mapping.values():

        ngram_to_freq = {''.join(int_to_ngram[index]): value for index, value in zip(range(len(fv)), fv)}

        fv_matrix.append(ngram_to_freq)

    hashed_matrix = h.transform(fv_matrix)

    return hashed_matrix


def select_prototypes(feature_matrix: csr_matrix, Pmax: float = 0.4):
    """Select prototypes from the matrix of hashed feature vectors.
    The referenced algorithm for selecting in approximately linear time:
    http://www.cs.columbia.edu/~verma/classes/uml/ref/clustering_minimize_intercluster_distance_gonzalez.pdf

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Feature matrix from hashed n-grams
    Pmax : float
        "Threshold for distances from data points to their nearest prototypes"
        default 0.4 (from paper)

    Returns
    -------
    (list (?), dict)
        List of selected prototypes. Perhaps we want to keep using a subset of the sparse matrix though?
        Mapping of prototypes to datapoints
    """
    pass


def cluster_prototypes(MinD: float = 0.5):
    pass


if __name__ == '__main__':
    main()
