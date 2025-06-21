import numpy as np
import pandas as pd

data = pd.read_csv('./datasets/mnist_train.csv')
data = np.array(data)
X, y = data[:, 1:], data[:, 0]
X = X / 255.0  # Normalize

np.random.seed(0)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs)  # He init
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    def backward(self, dvalues):
        # Not used directly, handled with combined softmax+crossentropy
        pass

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy:
    def forward(self, inputs, y_true):
        self.activation = Activation_Softmax()
        self.activation.forward(inputs)
        self.output = self.activation.output
        return Loss_CategoricalCrossentropy().forward(self.output, y_true)
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases
import matplotlib.pyplot as plt
import numpy as np

def predict(X, dense1, activation1, dense2, activation2, dense3, loss_activation):
    # Forward pass through the network
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    loss_activation.activation.forward(dense3.output)
    probs = loss_activation.activation.output
    predictions = np.argmax(probs, axis=1)
    return predictions, probs

def test_prediction(index, X, y, dense1, activation1, dense2, activation2, dense3, loss_activation):
    current_image = X[index]
    current_image_reshaped = current_image.reshape(1, -1)  # shape (1, 784)
    prediction, probs = predict(current_image_reshaped, dense1, activation1, dense2, activation2, dense3, loss_activation)
    print("Prediction:", prediction[0])
    print("Label:    ", y[index])

    plt.gray()
    plt.imshow(current_image.reshape(28, 28), interpolation='nearest')
    plt.title(f"Prediction: {prediction[0]}, Label: {y[index]}")
    plt.show()

def predict_batch(X, y, dense1, activation1, dense2, activation2, dense3, loss_activation):
    predictions, _ = predict(X, dense1, activation1, dense2, activation2, dense3, loss_activation)
    accuracy = np.mean(predictions == y)
    print(f"Batch accuracy: {accuracy:.4f}")
    return accuracy

def iterate_minibatches(X, y, batch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        excerpt = indices[start:end]
        yield X[excerpt], y[excerpt]

def main():
    dense1 = Layer_Dense(784, 128)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(128, 64)
    activation2 = Activation_ReLU()
    dense3 = Layer_Dense(64, 10)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_SGD(learning_rate=0.01)

    epochs = 20
    batch_size = 64
    for epoch in range(epochs):
        for X_batch, y_batch in iterate_minibatches(X, y, batch_size):
            dense1.forward(X_batch)
            activation1.forward(dense1.output)
            dense2.forward(activation1.output)
            activation2.forward(dense2.output)
            dense3.forward(activation2.output)
            loss = loss_activation.forward(dense3.output, y_batch)
            predictions = np.argmax(loss_activation.output, axis=1)
            accuracy = np.mean(predictions == y_batch)

            loss_activation.backward(loss_activation.output, y_batch)
            dense3.backward(loss_activation.dinputs)
            activation2.backward(dense3.dinputs)
            dense2.backward(activation2.dinputs)
            activation1.backward(dense2.dinputs)
            dense1.backward(activation1.dinputs)
            optimizer.update_params(dense1)
            optimizer.update_params(dense2)
            optimizer.update_params(dense3)
        print(f"epoch {epoch+1}: loss={np.mean(loss):.4f} acc={accuracy:.4f}")
    while True:
        index = int(input("Enter an index to test prediction (0-59999, or -1 to exit): "))
        if index == -1:
            break
        if 0 <= index < len(X):
            test_prediction(index, X, y, dense1, activation1, dense2, activation2, dense3, loss_activation)
        else:
            print("Index out of range. Please try again.")
if __name__ == "__main__":
    main()
