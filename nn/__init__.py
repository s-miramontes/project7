"""
BMI 203: Biocomputing Algorithms 
(Final) Project 7: Neural Network
"""

# importing methods and classes from py files
from .io import (read_text_file, read_fasta_file)
from .nn import NeuralNetwork
from .preprocess import (one_hot_encode_seqs, sample_seqs)

# creating the "first" version
__version__ = "0.0.1"