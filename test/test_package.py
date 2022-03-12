# BMI 203 Project 7: Neural Network

# Import necessary dependencies here
import numpy as np
from nn import preprocess
from nn import NeuralNetwork

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
    X = np.array([[1, 2, 3]])

    # get predictions, since we know what to expect
    pred = nn_.predict(X)

    actual = np.array([[55]])
    print(pred)

    #rounding bc python has invisible digits
    assert np.array_equal(pred, actual)


def test_binary_cross_entropy():
    """
    Checking whether bce implementation is correct, here the nn_
    dimensions are much smaller.
    Y and Y_pred are made up to test the accuracy of this
    method.
    """
    nn_ = NeuralNetwork(nn_arch = [{"input_dim": 3, "output_dim": 1, "activation": "relu"}],
                                   lr = 0.01,
                                   seed = 40,
                                   batch_size = 1,
                                   epochs = 1,
                                   loss_function = 'mse')

    # make up your own to test
    y = np.array([0, 1, 1, 1])
    y_pred = np.array([0.3, 0.2, 0.1, 0.4])

    bce = nn_._binary_cross_entropy(y, y_pred)

    # self-calculated, this is a horrible loss ha.
    assert np.isclose(bce, 1.29625)


def test_binary_cross_entropy_backprop():
    """
    Testing whether binary_cross_entropy_backprop executes
    as expected. Here the same network as above is created,
    just for simplicity. Inputs will remain the same as 
    above -- that is (y, and y_pred)
    """
    nn_ = NeuralNetwork(nn_arch = [{"input_dim": 3, "output_dim": 1, "activation": "relu"}],
                                   lr = 0.01,
                                   seed = 40,
                                   batch_size = 1,
                                   epochs = 1,
                                   loss_function = 'mse')
    # same as above
    y = np.array([0, 1, 1, 1])
    y_pred = np.array([0.3, 0.2, 0.1, 0.4])

    bce_back = nn_._binary_cross_entropy_backprop(y, y_pred)
    expected_bceb = np.array([ 0.35714286, -1.25, -2.5, -0.625])

    assert np.allclose(bce_back, expected_bceb)


def test_mean_squared_error():
    """
    Testing whether the output of mse is correct. We will be
    reusing the same network and y, y_pred here.
    """
    nn_ = NeuralNetwork(nn_arch = [{"input_dim": 3, "output_dim": 1, "activation": "relu"}],
                                   lr = 0.01,
                                   seed = 40,
                                   batch_size = 1,
                                   epochs = 1,
                                   loss_function = 'mse')

    # same as above
    y = np.array([0, 1, 1, 1])
    y_pred = np.array([0.3, 0.2, 0.1, 0.4])

    mce = nn_._mean_squared_error(y, y_pred)
    mce_actual = 0.475

    assert np.isclose(mce, mce_actual)


def test_mean_squared_error_backprop():
    """
    Testing whether mse backprop works correctly. Here we again use
    the same nn as above and inputs.
    """
    nn_ = NeuralNetwork(nn_arch = [{"input_dim": 3, "output_dim": 1, "activation": "relu"}],
                                   lr = 0.01,
                                   seed = 40,
                                   batch_size = 1,
                                   epochs = 1,
                                   loss_function = 'mse')

    # same as above
    y = np.array([0, 1, 1, 1])
    y_pred = np.array([0.3, 0.2, 0.1, 0.4])

    mce_back = nn_._mean_squared_error_backprop(y, y_pred)
    mce_back_actual = np.array([ 0.15, -0.4 , -0.45, -0.3 ])

    assert np.allclose(mce_back, mce_back_actual)


def test_one_hot_encode():
    """
    Here we test whether we can one-hot-encode a sequence
    correctly.
    """

    sequences = ["AGG", "TAC"]
    encoded_actual = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                              [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]])
    # perform one-hot-encode
    encoded_test = preprocess.one_hot_encode_seqs(sequences)

    # check they are equal
    assert np.array_equal(encoded_test, encoded_actual)


def test_sample_seqs():
    pass
