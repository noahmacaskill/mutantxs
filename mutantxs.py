"""MutantX-S Implementation for EMBER Function Imports

This script contains method stubs to guide Noah's planned
implementation of MutantX-S, which aims to be comparable
against COUGAR.
"""
import random
from json import loads
import numpy as np
from numpy import vstack
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from collections import OrderedDict


def main():
    info_list, record_list = open_ember_files()

    md5_to_ngrams = convert_function_imports_to_ngrams(info_list, record_list)

    md5_to_fvs, int_to_ngram = create_feature_vectors_from_ngrams(md5_to_ngrams)

    feature_matrix = reduce_dimensions_hashing_trick(md5_to_fvs)

    prototypes, protos_to_dps = select_prototypes(feature_matrix)

    clustered_prototypes = cluster_prototypes(feature_matrix, prototypes)


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
    md5_to_ngrams = dict()

    for md5, num_imports in info_list:

        md5_to_ngrams[md5] = list()

        for index in range(import_index, import_index + int(num_imports) - n + 1):
            n_gram = tuple([record[2].lower() + "," + record[1].lower() for record in record_list[index:index + n]])

            md5_to_ngrams[md5].append(n_gram)

        import_index += int(num_imports)

    return md5_to_ngrams


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
    md5_vector_mapping = OrderedDict()

    for k, v in sample_to_ngrams.items():
        md5_vector_mapping[k] = [0] * len(n_gram_encodings)

        for n_gram in v:
            md5_vector_mapping[k][encodings_reverse_dict[n_gram]] += 1

    return md5_vector_mapping, n_gram_encodings


def reduce_dimensions_hashing_trick(md5_vector_mapping: dict) -> csr_matrix:
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

    Returns
    -------
    X : sparse matrix of shape (n_samples, n_features)
        Feature matrix from hashed n-grams. We may densify this,
        depending on the memory requirements and ability to calculate
        Euclidean distance on sparse vectors.
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
    """

    h = FeatureHasher(2**12, input_type="string", alternate_sign=False)

    fv_matrix = list()

    for fv in md5_vector_mapping.values():
        indices = [str(i) for i in range(len(fv)) if fv[i] > 0]

        fv_matrix.append(indices)

    hashed_matrix = h.transform(fv_matrix)

    return hashed_matrix


def select_prototypes(feature_matrix: csr_matrix, Pmax: float = 0.4) -> tuple:
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

    prototypes = list()
    protos_to_dps = dict()

    # Randomly select first prototype
    prototypes.append(random.randint(0, feature_matrix.get_shape()[0] - 1))
    protos_to_dps[prototypes[0]] = [dp for dp in range(feature_matrix.get_shape()[0]) if dp != prototypes[0]]

    # Find next prototype using largest distance
    prototype_distances = normalize(pairwise_distances(feature_matrix.getrow(prototypes[0]), feature_matrix), norm="max")
    next_proto = np.argmax(prototype_distances)
    max_dist = np.max(prototype_distances)

    # Find new prototypes until all data points are within radius Pmax of a prototype
    while max_dist > Pmax and len(prototypes) < feature_matrix.get_shape()[0]:
        new_proto_distances = normalize(pairwise_distances(feature_matrix.getrow(next_proto), feature_matrix), norm="max")
        prototype_distances = vstack((prototype_distances, new_proto_distances))

        new_proto_dps = list()
        dps_to_remove = list()

        for prototype_index in range(len(prototypes)):

            prototype = prototypes[prototype_index]
            for datapoint in protos_to_dps[prototype]:
                if prototype_distances[prototype_index][datapoint] > prototype_distances[-1][datapoint]:

                    if next_proto != datapoint:
                        new_proto_dps.append(datapoint)

                    dps_to_remove.append(datapoint)

            protos_to_dps[prototype] = [dp for dp in protos_to_dps[prototype] if dp not in dps_to_remove]
            dps_to_remove.clear()

        # Create new prototype with corresponding cluster
        prototypes.append(next_proto)
        protos_to_dps[next_proto] = new_proto_dps

        max_dist = 0

        # Calculate next potential prototype as datapoint furthest from its current corresponding prototype
        for prototype_index in range(len(prototypes)):
            for datapoint in protos_to_dps[prototypes[prototype_index]]:
                distance_to_proto = prototype_distances[prototype_index][datapoint]

                if distance_to_proto > max_dist:
                    max_dist = distance_to_proto
                    next_proto = datapoint

    return prototypes, protos_to_dps


def cluster_prototypes(feature_matrix: csr_matrix, prototypes: list, MinD: float = 0.5) -> list:

    # Sort prototypes for simpler computations
    prototypes.sort()

    # Initialize clusters as singleton clusters for each prototype
    clusters = [[prototype] for prototype in prototypes]

    # Construct a distance matrix between prototypes
    prototype_to_prototype_distances = normalize(pairwise_distances(feature_matrix.getrow(prototypes[0]), feature_matrix), norm="max")

    for prototype in prototypes[1:]:
        prototype_distances = normalize(pairwise_distances(feature_matrix.getrow(prototype), feature_matrix), norm="max")
        prototype_to_prototype_distances = vstack((prototype_to_prototype_distances, prototype_distances))

    non_prototype_indices = [index for index in range(len(prototype_to_prototype_distances[0])) if index not in prototypes]
    prototype_to_prototype_distances = np.delete(prototype_to_prototype_distances, non_prototype_indices, axis=1)

    # Assign all zero distances (prototypes' distances to themselves) from the distance matrix to 2
    # where 2 is an arbitrary number > MinD
    prototype_to_prototype_distances[prototype_to_prototype_distances == 0] = 2

    # Compute minimum distance between two prototypes
    min_dist = prototype_to_prototype_distances.min()

    # Find clusters until the minimum distance between closest clusters is >= MinD
    while min_dist < MinD:
        indices = np.where(prototype_to_prototype_distances == min_dist)
        prototype1 = indices[0][0]
        prototype2 = indices[1][0]

        cluster_found = False
        new_cluster = list()

        # Combine prototype1 and prototype2 clusters together
        for cluster in clusters:
            if prototypes[prototype1] in cluster or prototypes[prototype2] in cluster:
                if not cluster_found:
                    new_cluster = cluster
                    cluster_found = True
                else:
                    for prototype in cluster:
                        new_cluster.append(prototype)
                    clusters.remove(cluster)

        # Assign distance between prototype1 and prototype2 to 2 where 2 is an arbitrary number > MinD
        prototype_to_prototype_distances[prototype1][prototype2] = 2
        prototype_to_prototype_distances[prototype2][prototype1] = 2

        min_dist = prototype_to_prototype_distances.min()

    return clusters


if __name__ == '__main__':
    main()
