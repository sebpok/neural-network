import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
import pickle
import scipy.ndimage
from show_grid import DrawGrid

# ----------- LAYERS -----------
class Layer:
    def forward(self, inputs):
        raise NotImplementedError
    def backward(self, dvalues):
        raise NotImplementedError
    def get_params(self):
        return []
    def set_params(self, params):
        pass

class Layer_Dense(Layer):
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
    def get_params(self):
        return [self.weights, self.biases]
    def set_params(self, params):
        self.weights, self.biases = params

class Activation_ReLU(Layer):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues * (self.inputs > 0)

class Activation_Softmax(Layer):
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    def backward(self, dvalues):
        # Usually used together with cross-entropy for efficiency
        pass

# ----------- LOSS -----------
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

class Activation_Softmax_Loss_CategoricalCrossentropy(Layer):
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.forward(self.output, y_true)
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

# ----------- OPTIMIZER -----------
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
    def update_params(self, layer):
        if hasattr(layer, 'weights'):
            layer.weights += -self.learning_rate * layer.dweights
            layer.biases += -self.learning_rate * layer.dbiases


# ----------- DATA AUGMENTATION UTILS -----------
def augment_image(img_flat):
    img = img_flat.reshape(28, 28)
    # Random shift
    shift = np.random.randint(-3, 4, 2)
    img = scipy.ndimage.shift(img, shift, mode='nearest')
    # Random rotation
    angle = np.random.uniform(-15, 15)
    img = scipy.ndimage.rotate(img, angle, reshape=False, mode='nearest')
    # Add gaussian noise
    img += np.random.normal(0, 0.05, img.shape)
    # Clip values to [0, 1]
    img = np.clip(img, 0, 1)
    return img.flatten()

# Optionally, preprocess user-drawn digits before prediction
def preprocess_user_image(img_flat):
    img = img_flat.reshape(28, 28)
    cy, cx = scipy.ndimage.center_of_mass(img)
    shiftx = np.round(img.shape[1] / 2.0 - cx).astype(int)
    shifty = np.round(img.shape[0] / 2.0 - cy).astype(int)
    img = scipy.ndimage.shift(img, (shifty, shiftx), mode='nearest')
    return img.flatten()

# ----------- NETWORK -----------
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def save(self, filename):
        params = [layer.get_params() for layer in self.layers]
        with open(filename, 'wb') as f:
            pickle.dump(params, f)
        print(f"Model saved to {filename}")

    def load(self, filename):
        with open(filename, 'rb') as f:
            params = pickle.load(f)
        for layer, layer_params in zip(self.layers, params):
            layer.set_params(layer_params)
        print(f"Model loaded from {filename}")


    def forward(self, X, training=True):
        output = X
        for layer in self.layers:
            if isinstance(layer, Activation_Softmax_Loss_CategoricalCrossentropy):
                break  # handled separately
            layer.forward(output)
            output = layer.output
        return output

    def predict(self, X):
        output = X
        probs = 0
        for layer in self.layers:
            if isinstance(layer, Activation_Softmax_Loss_CategoricalCrossentropy):
                layer.activation.forward(output)
                probs = layer.activation.output
                break
            layer.forward(output)
            output = layer.output
        print("----- PREDICTION ------")
        print(f"It's: {np.argmax(probs, axis=1)} | Probes: {probs}")
        return np.argmax(probs, axis=1), probs

    def train(self, X, y, epochs=10, batch_size=32, verbose=1):
        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled, y_shuffled = X[indices], y[indices]
            for start in range(0, X.shape[0], batch_size):
                end = start + batch_size
                X_batch, y_batch = X_shuffled[start:end], y_shuffled[start:end]
                
                # AUGMENTATION: augment each image in the batch
                # X_batch_aug = np.array([augment_image(x) for x in X_batch])

                # Forward pass
                output = X_batch
                for layer in self.layers:
                    if isinstance(layer, Activation_Softmax_Loss_CategoricalCrossentropy):
                        loss = layer.forward(output, y_batch)
                        break
                    layer.forward(output)
                    output = layer.output

                # Metrics
                predictions = np.argmax(layer.output, axis=1)
                accuracy = np.mean(predictions == y_batch)

                # Backward pass
                layer.backward(layer.output, y_batch)
                for l in reversed(self.layers[:-1]):
                    l.backward(self.layers[self.layers.index(l)+1].dinputs)

                # Update
                for l in self.layers:
                    if hasattr(l, 'weights'):
                        self.optimizer.update_params(l)
            if verbose:
                print(f"Epoch {epoch+1}: loss={np.mean(loss):.4f} acc={accuracy:.4f}")

    def evaluate(self, X, y):
        predictions, _ = self.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy

# ----------- DATA LOADING UTILS -----------
def load_data(mode, rows):
    if mode == 'test':
        data = pd.read_csv('./datasets/mnist_test.csv', nrows=rows)
        data = np.array(data)
        np.random.shuffle(data)
        data_test = data[:rows]
        X_train, y_train = 0, 0
        X_test, y_test = data_test[:, 1:] / 255.0, data_test[:, 0]
        return X_train, y_train, X_test, y_test.astype(int)

    elif mode == 'train':
        data = pd.read_csv('./datasets/mnist_train.csv', nrows=rows)
        data = np.array(data)
        np.random.shuffle(data)
        data_train = data[:rows]
        data_test = data[rows:rows+1000]
        X_train, y_train = data_train[:, 1:] / 255.0, data_train[:, 0]
        X_test, y_test = data_test[:, 1:] / 255.0, data_test[:, 0]
        return X_train, y_train.astype(int), X_test, y_test.astype(int)

    else:
        raise ValueError("Mode must be 'train' or 'test'")

# ------------- DATASET VISUALIZATION -----------
def show_image(image, prediction):
    plt.gray()
    plt.imshow(image.reshape(28, 28), interpolation='nearest')
    plt.title(f"Prediction: {prediction[0]}")
    plt.show()

def reshape_image_for_prediction(image):
    image = image.reshape(1, -1)  # Reshape to (1, 784)
    return image / 255.0  # Normalize

def main():
    X_train, y_train, X_test, y_test = load_data('test', 50)

    net = NeuralNetwork()
    net.add(Layer_Dense(784, 254))
    net.add(Activation_ReLU())
    net.add(Layer_Dense(254, 128))
    net.add(Activation_ReLU())
    net.add(Layer_Dense(128, 10))
    net.add(Activation_Softmax_Loss_CategoricalCrossentropy())
    net.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_SGD(learning_rate=0.05)
    )
    #net.train(X_train, y_train, epochs=200, batch_size=64)
    net.load("model_new.plk")
    net.evaluate(X_test, y_test)
    user_input = input("Enter '1' to show a random image, '2' to predict a drawn digit, or 'q' to quit: ")
    
    if user_input == '1':
        while True: 
            index = np.random.randint(0, len(X_test))
            image = X_test[index]
            prediction, _ = net.predict(image)
            show_image(image, prediction)
            if(input("Press Enter to show another image or 'q' to quit: ") == 'q'):
                break

    elif user_input == '2':
        root = tk.Tk()
        root.title("Rysowanie Paint 28x28 (ciągłe rozjaśnianie, z printem macierzy)")
        grid = DrawGrid(root)
        clear_btn = tk.Button(root, text="Wyczyść", command=grid.clear)
        clear_btn.pack()
        show_data_btn = tk.Button(root, text="Przewiduj", command=lambda: net.predict(preprocess_user_image(grid.data)))
        show_data_btn.pack()
        root.mainloop()


# ----------- USAGE -----------
if __name__ == "__main__":
    main()
