import numpy as np
import time

#load data
train_data = np.load("train_data.npy")
train_labels = np.load("train_label.npy")
test_data = np.load("test_data.npy")
test_labels = np.load("test_label.npy")

# Check shapes and data types
print("Train data shape:", train_data.shape)
print("Train labels shape:", train_labels.shape)
print("Train data type:", train_data.dtype)
print("Train labels type:", train_labels.dtype)

train_data = train_data / np.max(train_data)

def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

class Activation:

    def __tanh(self, x):
        return np.tanh(x)
    
    def __tanh_derivative(self, a):
        return 1.0 - a**2
    
    def __sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def __sigmoid_derivative(self, a):
        return a * (1 - a)
    
    def __relu(self, x):
        return np.maximum(0, x)
    
    def __relu_derivative(self, a):
        return np.heaviside(a, 0)

    def __leakyrelu(self, x, alpha=0.01):
        return np.where(x >= 0, x, alpha * x)
    
    def __leakyrelu_derivative(self, x, alpha=0.01):
        return np.heaviside(x, 1) * (1 - alpha) + alpha
    
    def __softmax(self, z):
        z = np.atleast_2d(z)
        max_z = np.max(z, axis=1, keepdims=True)
        z = z - max_z
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    
    def __softmax_derivative(self, z, z_hat):
        return z_hat - z
    
    def __init__(self, activation_function = 'relu'):

        if activation_function == "tanh":
            self.f = self.__tanh
            self.f_derivative = self.__tanh_derivative
        elif activation_function == "sigmoid":
            self.f = self.__sigmoid
            self.f_derivative = self.__sigmoid_derivative   
        elif activation_function == 'relu': 
            self.f = self.__relu
            self.f_derivative = self.__relu_derivative
        elif activation_function == 'leakyrelu':
            self.f = self.__leakyrelu
            self.f_derivative = self.__leakyrelu_derivative
        elif activation_function == "softmax":
            self.f = self.__softmax
            self.f_derivative = self.__softmax_derivative

class HiddenLayer():
    
    def __init__(self,
                 n_in,
                 n_out,
                 activation_function_previous_layer = 'relu',
                 activation_function = 'relu',
                 W = None,
                 b = None,
                 v_W = None, #Velcoity term
                 v_b = None, #Velocity of bias
                 last_hidden_layer = False):

        """
        T
        """
    
        # self.last_hidden_layer = last_hidden_layer


        self.input = None
        self.activation_function = Activation(activation_function).f

        # set activation derivative to derivative of activation function of layer that preceded current layer
        self.activation_function_derivative = None
        if activation_function_previous_layer:
            self.activation_function_derivative = Activation(activation_function_previous_layer).f_derivative
        
       
        #Xavier Initialisation - assign random small values (from uniform dist) for initial weights
        self.W = np.random.uniform(low = -np.sqrt(6. / (n_in + n_out)),
                                   high = np.sqrt(6. / (n_in + n_out )),
                                   size = (n_in, n_out))
       
        # Initialise all bias as 0
        self.b = np.zeros(n_out)

        # Sigmoid function is bounded by 0 <= sigma(x) <= 0.25
        if activation_function == 'sigmoid':
           self.W *= 4

        # Set the size of the weight and bias gradation as the size of weight and bias
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

        # self.v_W = np.zeros_like(self.grad_W)
        # self.v_b = np.zeros_like(self.grad_b)
        
        # self.binomial_array=np.zeros(n_out)
    
    def forward(self, input):
        
        # Compute linear output using I . W + b
        linear_output = np.dot(input, self.W) + self.b

        # Apply activation function to linear output (if any)
        self.output = (
            linear_output if self.activation_function is None
            else self.activation_function(linear_output)
        )

        # Store input for backpropagation
        self.input = input

        # Return output
        return self.output
    
    def backward(self, delta, layer_number, output_layer = False):

        # Compute how much each weight contributed to the error
        # derivate of the loss with respect to weights is equal to the input into that layer (output of layer 1 step closer to inputs) * delta of current layer
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        
        # Compute how much each weight contributed to the eroor
        self.grad_b = np.average(delta, axis=0)

        # From the above you know how much each input contributed to the error for this layer (steeper the gradient the more wrong it is so needs a bigger update)

        # Pass the error backward (find new delta)

        if self.activation_function_derivative:
            delta = delta.dot(self.W.T) * self.activation_function_derivative(self.input)

        return delta

class MLP:
    def __init__(self,
                 layers,
                 activation_function = [None, 'relu', 'relu','relu', 'softmax'],
                 weight_decay = 1.0):

        self.layers = []
        self.params = []

        self.activation_function = activation_function
        self.weight_decay = weight_decay

        for i in range(len(layers)-1):

            last_hidden_layer = False

            if i == len(layers) - 2: # -2 because -1 for output layer, and another -1 since it's index 0
                last_hidden_layer = True

            self.layers.append(HiddenLayer(layers[i], 
                                           layers[i+1], 
                                           activation_function[i], 
                                           activation_function[i+1],
                                           last_hidden_layer=last_hidden_layer))
    
    def forward(self, input):
        
        for layer in self.layers:
            output = layer.forward(input)

            input = output
        
        return output
    
    def CE_loss(self, z, z_hat):

        loss = - np.nansum(z * np.log(z_hat))
        loss = loss / z.shape[0]
        loss = loss * self.weight_decay
        delta = Activation(self.activation_function[-1]).f_derivative(z, z_hat)

        return loss, delta
    
    def backward(self, delta):

        delta = self.layers[-1].backward(delta, len(self.layers) -1, output_layer = True)
        for layer_number, layer in reversed(list(enumerate(self.layers[:-1]))):
            delta = layer.backward(delta, layer_number)
    
    def update(self, lr, SGD_optim):

        # if SGD_optim is None:
        for layer in self.layers:
            layer.W -= lr * layer.grad_W
            layer.b -= lr * layer.grad_b
    
    def fit(self, X, y, learning_rate = 0.1, epochs = 100, SGD_optim = None, batch_size = 1):
        X = np.array(X)
        y = np.array(y)
        training_loss = []
        training_accuracy = []
        testing_accuracy = []

        num_batches = int(np.ceil(X.shape[0] / batch_size))
            
        for k in range(epochs):

            loss = np.zeros(num_batches) 

            current_idx = 0 

            #Shuffle the data, to ensure that each epoch will have different sequence of observations
            X, y = Utils.shuffle(X, y)

            for batch_idx in range(num_batches):
                
                #forward pass 
                y_hat = self.forward(X[current_idx : current_idx + batch_size, :])

                #backward pass
                loss[batch_idx], delta = self.CE_loss(y[current_idx : current_idx + batch_size], y_hat)

                self.backward(delta)

                #update
                self.update(learning_rate, SGD_optim)

                #Update the index based on the batch window for the next round of Mini-Batch learning.
                if (current_idx + batch_size) > X.shape[0]:
                    batch_size = X.shape[0] - current_idx
                current_idx += batch_size

            #Predict and compute metrics for each run
            test_predict = self.predict(test_df.X)
            train_predict = self.predict(train_df.X)
            test_predict = test_df.decode(test_predict)
            train_predict = train_df.decode(train_predict)
            test_accuracy = np.sum(test_predict == test_labels[:, 0]) / test_predict.shape[0]
            train_accuracy = np.sum(train_predict == train_labels[:, 0]) / train_predict.shape[0]

            training_loss.append(np.mean(loss))
            training_accuracy.append(train_accuracy)
            testing_accuracy.append(test_accuracy)

            output_dict = {'Training Loss': training_loss, 'Training Accuracy': training_accuracy, 'Testing Accuracy': testing_accuracy}

            print(f'Epoch {k+1}/{epochs} has been trained with Train Loss: {str(round(training_loss[-1], 4))}, Training Accuracy: {str(round(training_accuracy[-1] * 100, 4))}% and Testing Accuracy: {str(round(testing_accuracy[-1] * 100, 4))}%.')
        
        return output_dict
    
    def predict(self, x):

        x = np.array(x)
        output = [i for i in range(x.shape[0])]
        for i in np.arange(x.shape[0]):
            output[i] = self.forward(x[i, :])
        output = np.array(output)
        return output

class Preprocessing:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.predictions = None

    def normalize(self):     

        norm_data = (self.X - np.min(self.X))/(np.max(self.X) - np.min(self.X))
        self.X = norm_data

    def standardize(self):

        self.X = (self.X - np.mean(self.X)) / np.std(self.X)

    @staticmethod
    def label_encode(label_vector):

        num_classes = np.unique(label_vector).size
        
        encoded_label_vector = []
        
        for label in label_vector:
            encoded_label = np.zeros(num_classes)
            encoded_label[int(label)] = 1
            encoded_label_vector.append(encoded_label)
        
        encoded_label_vector = np.array(encoded_label_vector) 

        return encoded_label_vector
    
    @staticmethod
    def decode(prediction_matrix):

        decoded_predictions = np.zeros(prediction_matrix.shape[0])
        for prediction_idx, prediction_vector in enumerate(prediction_matrix):
            decoded_predictions[prediction_idx] = int(np.argmax(prediction_vector)) # we add the two index zeros because it's a nparray within a tuple
        
        return decoded_predictions

class Utils:

    @staticmethod
    def shuffle(X, y):

        shuffled_idx = np.arange(X.shape[0])
        np.random.shuffle(shuffled_idx)
        X = X[shuffled_idx]
        y = y[shuffled_idx]

        return X, y


# Instantiating our data and pre-processing it as required
train_df = Preprocessing(train_data, train_labels)
test_df = Preprocessing(test_data, test_labels)

# train_df.normalize()
# test_df.normalize()
train_df.standardize()
test_df.standardize()

# Perform one-hot encoding for our label vector (ONLY ON TRAIN)

train_df.y = train_df.label_encode(train_df.y)



LAYER_NEURONS = [128, 120, 60, 10]
LAYER_ACTIVATION_FUNCS = [None, 'leakyrelu', 'leakyrelu', 'softmax']
LEARNING_RATE = 0.005
EPOCHS = 20
DROPOUT_PROB = 0.5 
SGD_OPTIM = None
BATCH_SIZE = 250
WEIGHT_DECAY = 1.0
# Instantiate the multi-layer neural network
nn = MLP(LAYER_NEURONS, LAYER_ACTIVATION_FUNCS, weight_decay = WEIGHT_DECAY)

t0 = time.time()
trial1 = nn.fit(train_df.X, train_df.y, learning_rate = LEARNING_RATE, epochs = EPOCHS, SGD_optim = SGD_OPTIM, batch_size=BATCH_SIZE )
t1 = time.time()
print(f"============= Model Build Done =============")
print(f"Time taken to build model: {round(t1 - t0, 4)} seconds.")

    

                



