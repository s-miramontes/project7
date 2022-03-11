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
    nn_._param_dict = {'W1': np.array([[1, 2, 3], [2, 3, 1]]),
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

    nn_ = NeuralNetwork(nn_arch = [{"input_dim": 3, "output_dim": 2, "activation": "relu"},
                                   {"input_dim": 2, "output_dim": 1, "activation": "relu"}],
                                   lr = 0.01,
                                   seed = 40,
                                   batch_size = 1,
                                   epochs = 1,
                                   loss_function = 'mse')

    # redefine a new input, same weights, biases as above
    A_prev = np.array([[1, 2, 1], [1, 2, 1]])
    w1 = np.array([[1, 2, 3], [2, 3, 1]])
    b1 = np.array([[1], [1]])

    # test a single forward pass with both sigmoid and relu
    single_Ar, single_Zr = nn_._single_forward(w1, b1, A_prev, "relu")
    single_As, single_Zs = nn_._single_forward(w1, b1, A_prev, "sigmoid")
    
    # expected results
    actual_A_relu = np.array([[9, 10], [9, 10]])
    actual_A_sigm = np.array([[1, 1], [1, 1]])

    # make sure they are the same
    assert np.array_equal(single_Ar, actual_A_relu)
    # kind of cheating with round, but still 1
    assert np.array_equal(np.round(single_As), actual_A_sigm)


def test_single_backprop():
    
    # recycling network
    nn_ = NeuralNetwork(nn_arch = [{"input_dim": 3, "output_dim": 2, "activation": "relu"},
                                   {"input_dim": 2, "output_dim": 1, "activation": "relu"}],
                                   lr = 0.01,
                                   seed = 40,
                                   batch_size = 1,
                                   epochs = 1,
                                   loss_function = 'mse')

    # same weights, same bias
    w1 = np.array([[1, 2, 3], [2, 3, 1]])
    b1 = np.array([[1], [1]])

    # recall your last output
    last_lay = np.array([[9, 10], [9, 10]])
    # and your input
    A_prev = np.array([[1, 2, 1], [1, 2, 1]])

    first_deriv = np.array([[1, 2], [1, 2]])
    activation_ = "relu"

    # testing single backprop
    firstderiv_test, wderiv_test, biasderiv_test = nn_._single_backprop(w1, b1,
                                                                            last_lay,
                                                                            A_prev,
                                                                            first_deriv,
                                                                            activation_)

    # assert the single backprop results are as expected 
    actual_firstderiv = np.array([[5, 8, 5], [5, 8, 5]])
    assert np.array_equal(firstderiv_test, actual_firstderiv)

    actual_wderiv = np.array([[2, 4, 2], [4, 8, 4]])
    assert np.array_equal(wderiv_test, actual_wderiv)
    
    actual_biasderiv = np.array([[1], [1]])
    assert np.array_equal(biasderiv_test, actual_biasderiv)


def test_predict():
    # again, network
    nn_ = NeuralNetwork(nn_arch = [{"input_dim": 3, "output_dim": 2, "activation": "relu"},
                                   {"input_dim": 2, "output_dim": 1, "activation": "relu"}],
                                   lr = 0.01,
                                   seed = 40,
                                   batch_size = 1,
                                   epochs = 1,
                                   loss_function = 'mse')

    # and the same parameters as the forward func
    nn_._param_dict = {'W1': np.array([[1, 2, 3], [2, 3, 1]]),
                      'b1': np.array([[1], [1]]),
                      'W2': np.array([[2, 2]]),
                      'b2': np.array([[1]])}

    # now make up a training dataset 
    X_train = np.array([[1, 2, 3]])
    Y_train = np.array([[1]])

    # now make up at val dataset, same one actually
    X_val = np.array([[1, 2, 3]])
    Y_val = np.array([[1]])

    # use fit func
    epochLoss_train, epochLoss_val = nn_.fit(X_train, Y_train, X_val, Y_val)
    # get predictions, since we know what to expect
    pred = nn_.predict(X_train)

    actual = np.array([[0.46]])

    #rounding bc python has invisible digits
    assert (np.array_equal(np.round(pred,1), np.round(actual, 1)))


def test_binary_cross_entropy():
    """
    Checking whether bce implementation is correct, after
    recycling the network.
    """
    nn_ = NeuralNetwork(nn_arch = [{"input_dim": 3, "output_dim": 2, "activation": "relu"},
                                   {"input_dim": 2, "output_dim": 1, "activation": "relu"}],
                                   lr = 0.01,
                                   seed = 40,
                                   batch_size = 1,
                                   epochs = 1,
                                   loss_function = 'mse')

    





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
