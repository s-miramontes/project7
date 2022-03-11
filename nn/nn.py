# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Dict, Tuple
from numpy.typing import ArrayLike


# Neural Network Class Definition
class NeuralNetwork:
    """
    This is a neural network class that generates a fully connected Neural Network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            This list of dictionaries describes the fully connected layers of the artificial neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32}, {'input_dim': 32, 'output_dim': 8}] will generate a
            2 layer deep fully connected network with an input dimension of 64, a 32 dimension hidden layer
            and an 8 dimensional output.
        lr: float
            Learning Rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            This list of dictionaries describing the fully connected layers of the artificial neural network.
    """
    def __init__(self,
                 nn_arch: List[Dict[str, int]],
                 lr: float,
                 seed: int,
                 batch_size: int,
                 epochs: int,
                 loss_function: str):
        # Saving architecture
        self.arch = nn_arch
        # Saving hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs # num of epochs 
        self._loss_func = loss_function
        self._batch_size = batch_size
        # Initializing the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD!! IT IS ALREADY COMPLETE!!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """
        # seeding numpy random
        np.random.seed(self._seed)
        # defining parameter dictionary
        param_dict = {}
        # initializing all layers in the NN
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            # initializing weight matrices
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            # initializing bias matrices
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1
        return param_dict

    def _single_forward(self,
                        W_curr: ArrayLike,
                        b_curr: ArrayLike,
                        A_prev: ArrayLike,
                        activation: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """

        # Calculate Z_curr 
        # Z_curr is dot product of prev layer and weights + bias
        Z_curr = np.dot(A_prev, W_curr.T) + b_curr.T

        # Apply chose activation functions
        if activation =="relu":
            # ReLu on Z_curr
            A_curr = self._relu(Z_curr)

        elif activation == 'sigmoid':
            # sigmoid on Z_curr
            A_curr = self._sigmoid(Z_curr)

        return A_curr, Z_curr
        

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.
        Using single_forward throughout.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        # define cache dict: Z and A storage
        cache = {}

        # fill appropriately @ start
        cache['A0'] = X
        # A_prev is input X @ start
        A_prev = X

        # now tackle all layers
        for i, layer in enumerate(self.arch):
            # keep track of indices (note we already have 0)
            idxLayer = i + 1

            # define weights and biases outside of call 
            weight_curr = self._param_dict['W' + str(idxLayer)]
            bias_curr = self._param_dict['b' + str(idxLayer)]
            
            # move forward with defined weights, biases, and A_"prev"
            A_curr, Z_curr = self._single_forward(weight_curr, bias_curr,
                                                 A_prev, layer['activation'])

            # add A + Z to cache dict
            cache['A' + str(idxLayer)] = A_curr 
            cache['Z' + str(idxLayer)] = Z_curr 

            # redefine A_prev to be recently calculated A_curr
            A_prev = A_curr

            # ...repeat until layers are over

        # when done, return A_curr + cache (your history)
        return A_curr, cache


    def _single_backprop(self,
                         W_curr: ArrayLike,
                         b_curr: ArrayLike,
                         Z_curr: ArrayLike,
                         A_prev: ArrayLike,
                         dA_curr: ArrayLike,
                         activation_curr: str) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        # get dims of A
        A_dims = A_prev.shape[1]

        # backpropagation according to activation function
        # dZ_curr: deriv of activation function applied to Z_curr
        if activation_curr == 'relu':
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)
        elif activation_curr == 'sigmoid':
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)

        # partial derivs all the way to the front
        dA_prev = np.dot(dZ_curr, W_curr)
        dW_curr = np.dot(dZ_curr.T, A_prev)
        db_curr = np.sum(dZ_curr, axis = 1, keepdims = True) / A_dims

        return dA_prev, dW_curr, db_curr


    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.
        Using the single_backprop method.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """

        # define grad_dict
        grad_dict = {}

        # dA_curr from the selected loss function
        if self._loss_func == 'mse':
            dA_curr = self._mean_squared_error_backprop(y, y_hat)
        elif self._loss_func == 'bce':
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat)

        for i, layer in reversed(list(enumerate(self.arch))):
            # keep track of indices
            idxLayer = i + 1

            # weights and bias of current layer
            weight_curr = self._param_dict['W' + str(idxLayer)]
            bias_curr = self._param_dict['b' + str(idxLayer)]
            
            # get the transformed inputs of curr layer from cache 
            Z_curr = cache['Z' + str(idxLayer)]
            # get inputs of the prev layer from cache, not curr idx
            A_prev = cache['A' + str(idxLayer - 1)]

            # single backpropagation in this layer, using info above
            dA_prev, dW_curr, db_curr = self._single_backprop(weight_curr,
                                        bias_curr, Z_curr, A_prev, dA_curr,
                                        layer['activation'])

            # add gradients to dictionary
            grad_dict['dW' + str(idxLayer)] = dW_curr
            grad_dict['db' + str(idxLayer)] = db_curr

            # dA_curr is now the dA_prev (moving backwards)
            dA_curr = dA_prev

            #...repeat until done

        return grad_dict


    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and thus does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.

        Returns:
            None
        """
        for i, layer in enumerate(self.arch):
            # keep track of index
            idxLayer = i + 1

            # changing weight parameters appropriately -neg lr * gradient dict value of dW and db
            self._param_dict['W' + str(idxLayer)] -= self._lr * grad_dict['dW' + str(idxLayer)]
            # same as above, but updating biases
            self._param_dict['b' + str(idxLayer)] -=self._lr * grad_dict['db' + str(idxLayer)]

        return None


    def fit(self,
            X_train: ArrayLike,
            y_train: ArrayLike,
            X_val: ArrayLike,
            y_val: ArrayLike) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network via training for the number of epochs defined at
        the initialization of this class instance.
        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """

        # init list containing epoch loss: train + val
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        # iterate num epochs of times
        for e in range(epochs):

            # combine both X and y train to shuffle -- good practice
            shuffle_inputs = np.concatenate([X_train, y_train], axis=1)
            np.random.shuffle(shuffle_inputs)

            # X and y train redefined
            X_train = shuffle_inputs[:, :-1]
            y_train = shuffle_inputs[:, -1:].flatten()

            # batch size relies on total inputs
            n_batches = np.ceil(X_train.shape[0]/self._batch_size)
            # cut off according to determined batces
            X_batch = np.array_split(X_train, n_batches)
            y_batch = np.array_split(y_train, n_batches)

            # now that batches are defined, iterate until epoch is done
            # create lists to keep track of loss in train and val sets
            epoch_loss_train = []
            epoch_loss_val = []
            for X, y in zip(X_batch, y_batch):

                # get y_hat
                out, cache = self.forward(X)

                # done with forward, calculate loss
                if self._loss_func == 'mse':
                    train_loss = self._mean_squared_error(y, out)
                elif self._loss_func == 'bce':
                    train_loss = self._binary_cross_entropy(y, out)

                # add to the lists in this epoch
                epoch_loss_train.append(train_loss)

                # backward pass
                grad_dict = self.backprop(y, out, cache)
                self._update_params(grad_dict)

                # Validation
                out_val = self.predict(X_val)

                # now validation loss
                if self._loss_func == 'mse':
                    val_loss = self._mean_squared_error(y_val, out_val)
                elif self._loss_func == 'bce':
                    val_loss = self._binary_cross_entropy(y_val, out_val)

                # now add loss to the validation list
                epoch_loss_val.append(val_loss)

            # add the avg loss per epoch to the list
            per_epoch_loss_train.append(np.mean(epoch_loss_train))
            per_epoch_loss_val.append(np.mean(epoch_loss_val))

        return per_epoch_loss_train, per_epoch_loss_val


    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network model.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        
        # a prediction requires a forward pass to get y_hat
        # note we don't need cache here
        y_hat, _ = self.forward(X)

        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """

        # calculate sigmoid with Z
        nl_transform = 1 / (1 + np.exp(-Z))

        return nl_transform

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """

        # ReLu is max of 0 and input
        nl_transform = np.maximum(0, Z)

        return nl_transform

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """


        # Chain Rule
        # dA = dL/dA; 
        # dL/dZ = dL/dA * dA/dZ (deriv of sigmoid wrt Z)
        dZ = dA * self._sigmoid(Z) * (1 - self._sigmoid(Z))

        return dZ
        

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        
        # apply ReLu to Z, establish boundaries for <0 and 1
        Z = np.where(Z<0, 0, 1)
        dZ = Z * dA # dL/dA = dA, dA/dZ = Z rectified

        return dZ


    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        # loss function
        loss = -np.mean((y * np.log(y_hat)) + ((1 - y) * (np.log(1 - y_hat))))

        return loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # total number of terms in y
        y_length = y.shape[0]

        # deriv of loss wrt y hat
        dA = (((1 - y)/(1 - y_hat)) - (y/y_hat)) / y_length

        return dA


    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """

        # mse: (observed - predicted)**squared
        # and avg
        loss = np.mean((y - y_hat) ** 2)

        return loss


    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        
        y_length = y.shape[0]
        # derivative
        dA = 2 * (y_hat - y) / y_length

        return dA


    # adding code here in the case user wants to pick either bce or mse for loss?...maybe
    def _loss_function(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Loss function, computes loss given y_hat and y. This function is
        here for the case where someone would want to write more loss
        functions than just binary cross entropy.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.
        Returns:
            loss: float
                Average loss of mini-batch.
        """
        pass

    def _loss_function_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        This function performs the derivative of the loss function with respect
        to the loss itself.
        Args:
            y (array-like): Ground truth output.
            y_hat (array-like): Predicted output.
        Returns:
            dA (array-like): partial derivative of loss with respect
                to A matrix.
        """
        pass
