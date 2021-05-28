"""MutantX-S Implementation for EMBER Function Imports

This script contains method stubs to guide Noah's planned
implementation of MutantX-S, which aims to be comparable
against COUGAR.
"""
import random
from json import loads
from sys import argv
from collections import OrderedDict, Counter
import logging as lg

import numpy as np
from numpy import vstack
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import pairwise_distances, f1_score, homogeneity_completeness_v_measure, precision_recall_fscore_support
from sklearn.preprocessing import normalize
from tqdm import tqdm


def main():
    file_names = None
    if len(argv) > 1:
        file_names = argv[1:]

    info_list, record_list, md5_to_avclass = open_ember_files(file_names)

    lg.info('Loaded records from {} samples, converting to N-Grams...'.format(len(info_list)))
    md5_to_ngrams = convert_function_imports_to_ngrams(info_list, record_list)

    lg.info('Converting to feature vectors...')
    md5_to_fvs, int_to_ngram = create_feature_vectors_from_ngrams(md5_to_ngrams)

    lg.info('Hashing feature vectors...')
    feature_matrix = reduce_dimensions_hashing_trick(md5_to_fvs)

    lg.info('Selecting prototypes...')
    prototypes, prototypes_to_data_points = select_prototypes(feature_matrix)

    lg.info('Clustering {} prototypes...'.format(len(prototypes)))
    clustered_prototypes = cluster_prototypes(feature_matrix, prototypes)

    lg.info('Converting indices back to MD5s...')
    md5_clusters, md5_prototype_clusters = indices_to_md5s(clustered_prototypes, prototypes_to_data_points, list(md5_to_fvs.keys()))

    lg.info('Scoring clustering...')
    precision, recall, fscore_macro, fscore_micro, fscore_weighted, homogeneity, completeness, v_measure = \
        score_clustering(md5_clusters, md5_prototype_clusters, md5_to_avclass)

    lg.info('Creating signatures for each cluster...')
    signatures = cluster_signatures(int_to_ngram, md5_clusters, md5_to_fvs)
    lg.info('Done!')


def open_ember_files(file_names: list = None) -> tuple:
    """
    Import required information from EMBER data

    Returns
    -------
    (list, list)
        list of md5s, their respective number of function imports, and their AVClass labeling
        list of information on each function imports
    """

    md5_file = open("10K.md5", 'r')
    md5s = list()

    for line in md5_file:
        md5s.append(line[:-1])

    md5_file.close()

    info_list = list()
    record_list = list()
    md5_to_avclass = dict()

    if file_names is None:
        file_names = input("Select file names separated by spaces: ")
        file_names = file_names.split()

    # Import required information from each file
    for file_name in tqdm(file_names, desc='LoadFiles'):

        lg.info('Loading records from file: {}'.format(file_name))
        with open(file_name, 'r') as f:

            # Import required information from each malware sample (lind of file)
            for line in f:
                json_doc = loads(line)

                md5 = json_doc['md5']

                if md5 in md5s:

                    imports = json_doc['imports']
                    avclass = json_doc['avclass']
                    one_sample_records = list()

                    count = 0
                    for library in imports:
                        for function in imports[library]:
                            count += 1
                            one_sample_records.append((md5, library, function))

                    if count >= 4:
                        info_list.append((md5, str(count)))
                        record_list.extend(one_sample_records.copy())
                        one_sample_records.clear()
                        md5_to_avclass[md5] = avclass

    return info_list, record_list, md5_to_avclass


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
    md5_to_ngrams = OrderedDict()

    # Iterate over function imports for each malware sample, creating N-grams along the way
    for md5, num_imports in tqdm(info_list, desc='MakeN-Grams'):

        md5_to_ngrams[md5] = list()

        for index in range(import_index, import_index + int(num_imports) - n + 1):
            ngram = tuple([record[2].lower() + "," + record[1].lower() for record in record_list[index:index + n]])

            md5_to_ngrams[md5].append(ngram)

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
    ngrams = {ngram for ngram_list in sample_to_ngrams.values() for ngram in ngram_list}

    # Create a unique numerical encoding for each observed N-gram
    ngram_encodings = {encoding: ngram for encoding, ngram in zip(range(len(ngrams)), ngrams)}

    # Create a reverse dictionary from N-grams to encodings
    encodings_reverse_dict = {v: k for k, v in ngram_encodings.items()}

    # Create feature vectors to represent each sample
    md5_vector_mapping = OrderedDict()

    for md5, ngram_list in tqdm(sample_to_ngrams.items(), desc='CreateFVs'):
        md5_vector_mapping[md5] = [0] * len(ngram_encodings)

        for ngram in ngram_list:
            md5_vector_mapping[md5][encodings_reverse_dict[ngram]] += 1

    return md5_vector_mapping, ngram_encodings


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

    # Hash each feature vector to a smaller dimension using the sklearn feature hasher
    h = FeatureHasher(2**12, input_type="string", alternate_sign=False)

    fv_matrix = list()

    for fv in tqdm(md5_vector_mapping.values(), desc='BuildStrFVs'):
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
    prototypes_to_data_points = dict()

    # Randomly select first prototype
    prototypes.append(random.randint(0, feature_matrix.get_shape()[0] - 1))
    data_points = [data_point for data_point in range(feature_matrix.get_shape()[0]) if data_point != prototypes[0]]
    prototypes_to_data_points[prototypes[0]] = data_points

    # Find next prototype using largest distance
    prototype_distances = normalize(pairwise_distances(feature_matrix.getrow(prototypes[0]), feature_matrix), norm="max")
    next_potential_prototype = np.argmax(prototype_distances)
    max_dist = np.max(prototype_distances)

    # Find new prototypes until all data points are within radius Pmax of a prototype
    while max_dist > Pmax and len(prototypes) < feature_matrix.get_shape()[0]:
        new_prototype = next_potential_prototype

        new_prototype_distances = normalize(pairwise_distances(feature_matrix.getrow(new_prototype), feature_matrix), norm="max")
        prototype_distances = vstack((prototype_distances, new_prototype_distances))

        new_prototype_data_points = list()
        data_points_to_remove = list()

        # For each datapoint, determine whether it needs to be shifted to the new prototype, while also
        # keeping track of that max distance between data points and their closest prototypes to determine
        # the next potential prototype
        max_dist = 0

        for prototype_index in range(len(prototypes)):

            prototype = prototypes[prototype_index]
            for data_point in prototypes_to_data_points[prototype]:
                distance_to_current_prototype = prototype_distances[prototype_index][data_point]
                distance_to_new_prototype = prototype_distances[-1][data_point]

                # If data point is closer to new prototype, move it to new cluster
                if distance_to_current_prototype > distance_to_new_prototype:

                    if new_prototype != data_point:
                        new_prototype_data_points.append(data_point)

                    data_points_to_remove.append(data_point)

                    distance_to_current_prototype = distance_to_new_prototype

                # If a new max distance to a datapoint's closest prototype is found
                # update the max distance and the next potential prototype
                if distance_to_current_prototype > max_dist:
                    max_dist = distance_to_current_prototype
                    next_potential_prototype = data_point

            # Remove data points from current cluster that are being moved to new cluster
            prototypes_to_data_points[prototype] = [data_point for data_point in prototypes_to_data_points[prototype]
                                                    if data_point not in data_points_to_remove]
            data_points_to_remove.clear()

        # Create new prototype with corresponding cluster
        prototypes.append(new_prototype)

        prototypes_to_data_points[new_prototype] = new_prototype_data_points

    return prototypes, prototypes_to_data_points


def cluster_prototypes(feature_matrix: csr_matrix, prototypes: list, MinD: float = 0.5) -> list:
    """
    Clusters prototypes together such that no two prototypes are within a certain threshold of one another
    Parameters
    ----------
    feature_matrix: sparse matrix of shape (n_samples, n_features)
        Feature matrix from hashed n-grams
    prototypes: list
        List of prototype row indices in the feature matrix
    MinD: float
        Distance threshold for minimum distance between prototypes of a cluster

    Returns
    -------
    list:
        list of lists representing clusters
    """

    # Sort prototypes for simpler computations
    prototypes.sort()

    # Initialize clusters as singleton clusters for each prototype
    clusters = [[prototype] for prototype in prototypes]

    feature_matrix = feature_matrix[prototypes]

    prototype_to_prototype_distances = normalize(pairwise_distances(feature_matrix, feature_matrix), norm="max")

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


def indices_to_md5s(prototype_clusters: list, prototypes_to_data_points: dict, md5s: list) -> tuple:
    """
    Groups clusters together using their original md5s
    Parameters
    ----------
    prototype_clusters: list
        List of lists representing clustered prototypes
    prototypes_to_data_points: dict
        Dict mapping from each prototype to the data points within their cluster
    md5s: list
        List of all md5s

    Returns
    -------
    (List, List)
        A list of lists where each inner list contains md5s representing a cluster
        A list of lists where each inner lists contains md5s representing prototypes within a cluster
    """
    md5_clusters = list()
    current_cluster = list()

    md5_prototype_clusters = list()
    current_prototype_cluster = list()

    for cluster in prototype_clusters:
        for prototype in cluster:
            current_cluster.append(md5s[prototype])
            current_prototype_cluster.append(md5s[prototype])

            for data_point in prototypes_to_data_points[prototype]:
                current_cluster.append(md5s[data_point])

        md5_clusters.append(current_cluster.copy())
        current_cluster.clear()

        md5_prototype_clusters.append(current_prototype_cluster.copy())
        current_prototype_cluster.clear()

    return md5_clusters, md5_prototype_clusters


def score_clustering(md5_clusters: list, prototype_clusters: list, md5_to_avclass: dict):
    """
    Scores clustering by various metrics (precision, recall, F-scores, homogeneity, completeness, V-Measure)
    Parameters
    ----------
    md5_clusters: list
        List of lists of md5s, where each inner list represents a cluster
    prototype_clusters: list
        List of lists of md5s, where each inner list represents a cluster of prototypes
    md5_to_avclass: dict
        Mapping from md5s to AVClass labels attributed to each malware sample

    Returns
    -------
    Tuple containing all scoring metrics
    """

    y_true = list()
    y_pred = list()

    # Assign cluster label as most common AVClass labelling among prototypes in a cluster
    for cluster_index in range(len(prototype_clusters)):
        classes = [md5_to_avclass[md5] for md5 in prototype_clusters[cluster_index]]

        class_count = Counter(classes)
        cluster_label = class_count.most_common(1)[0][0]

        # Assign predicted and true (AVClass) labels to each sample
        for md5 in md5_clusters[cluster_index]:
            y_true.append(md5_to_avclass[md5])
            y_pred.append(cluster_label)

    # Score clustering
    precision, recall, fscore_macro, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='macro')
    fscore_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    fscore_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(labels_true=y_true, labels_pred=y_pred)

    return precision, recall, fscore_macro, fscore_micro, fscore_weighted, homogeneity, completeness, v_measure


def cluster_signatures(int_to_ngram: dict, md5_clusters: list, md5_to_fvs: dict):
    """
    Creates cluster signatures based on shared N-grams between elements in a cluster
    Parameters
    ----------
    int_to_ngram: dict
        Integer encoding for each N-gram corresponding to index in feature vector
    md5_clusters: list
        List of lists
    md5_to_fvs: dict
        Mapping from md5s to feature vectors

    Returns
    -------
    List
        List of signatures (N-grams) corresponding to each cluster
    """

    signatures = list()
    features = list()

    for cluster in md5_clusters:

        for md5 in cluster:
            features.extend([index for index in range(len(md5_to_fvs[md5])) if md5_to_fvs[md5][index] > 0])

        counter = Counter(features)

        if len(counter) >= 7:
            common_ngrams = counter.most_common(7)
            signature = [int_to_ngram[ngram[0]] for ngram in common_ngrams]
        else:
            signature = [int_to_ngram[ngram[0]] for ngram in counter]

        signatures.append(signature)

        features.clear()

    return signatures


if __name__ == '__main__':
    lg.basicConfig(
        format='[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=lg.INFO
    )
    main()
