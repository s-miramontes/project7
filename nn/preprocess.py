# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike


# Defining preprocessing functions
def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one hot encoding of a list of nucleic acid sequences
    for use as input into a fully connected neural net.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence
            length due to the one hot encoding scheme for nucleic acids.

            For example, if we encode 
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    """
    dict_seqs = {'A':[1, 0, 0, 0],
                 'T':[0, 1, 0, 0],
                 'C':[0, 0, 1, 0],
                 'G':[0, 0, 0, 1]}

    # for saving the 1-hot encoded sequence
    encodings = []

    # iterating through sequences in seq_arr
    for seq in seq_arr:

        # for storing encodings 
        oh_coded = []
        for letter in seq:
            # given 'letter', add the values in dict_seqs
            oh_coded += dict_seqs[letter]
        
        # add the one hot encoded sequence to encoding list
        encodings += [oh_coded]
    
    # since asked for array, convert
    return np.array(encodings)


def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample your sequences to account for class imbalance. 
    Consider this as a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # total num of labels
    n_labels = len(labels)
    # since we want equal amount of class representation assume 50/50 split
    size = np.ceil(n_labels/2)

    # zip labels and sequences
    seq_labels = list(zip(seqs, labels))

    # get pos and neg labels
    pos_labels = [el[0] for el in seq_labels if el[1] == True]
    neg_labels = [el[0] for el in seq_labels if el[0] == False]

    # resample from uniform distribution 
    pos_sample = np.random.choice(pos_labels, size=size)
    pos_sample = [(el, True) for el in pos_sample]
    neg_sample = np.random.choice(neg_labels, size=size)
    neg_sample = [(el, False) for el in neg_sample]
    all_ = pos_sample + neg_sample
    np.random.shuffle(all_)

    sampled_sequences = [j[0] for j in all_]
    sampled_labels = [j[1] for j in all_]

    return sampled_sequences, sampled_labels

