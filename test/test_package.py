# BMI 203 Project 7: Neural Network

# Import necessary dependencies here
import numpy as np
from nn import preprocess
from nn import NeuralNetwork

# TODO: Write your test functions and associated docstrings below -- uuughh

# create instance of neural network, so code isn't repeaded

def test_forward():
    nn_ = NeuralNetwork(nn_arch = [{"input_dim": 3, "output_dim": 2, "activation": "relu"},
                                   {"input_dim": 2, "output_dim": 1, "activation": "relu"}],
                                   lr = 0.01,
                                   seed = 40,
                                   batch_size = 1,
                                   epochs = 1,
                                   loss_function = 'mse')

    # establish weights and biases
    forward_params = {'W1': np.array([[1, 2, 3], [2, 3, 1]]),
                      'b1': np.array([[1], [1]]),
                      'W2': np.array([[2, 2]]),
                      'b2': np.array([[1]])}
   

    # your prev layer, or input
    Aprev = np.array([1, 2, 1])

    # pass Aprev/input into network
    result, cash = nn_.forward(Aprev)
    expected_result = 39

    # make sure cache is storing the result of operations, and output
    assert np.array_equal(cash['A0'], Aprev)
    assert np.array_equal(cash['A1'], np.array([[9, 10]]))
    assert np.array_equal(cash['Z1'], np.array([[9, 10]]))
    assert np.array_equal(cash['A2'], np.array([[39]]))
    assert np.array_equal(cash['Z2'], np.array([[39]]))
    assert result == expected_result


def test_single_forward():
    pass


def test_single_backprop():
    pass


def test_predict():
    pass


def test_binary_cross_entropy():
    pass


def test_binary_cross_entropy_backprop():
    pass


def test_mean_squared_error():
    pass


def test_mean_squared_error_backprop():
    pass


def test_one_hot_encode():
    pass


def test_sample_seqs():
    pass
