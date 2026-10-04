import random
import math
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Network Parameters
global activation_fuction
activation_fuction = "LEAKY RELU"
learning_rate = 0.0025
global total_loss


def activation(value,output):
    if output:
        return value
    if activation_fuction == "RELU":
        if value < 0:
            value = 0

    elif activation_fuction == "LEAKY RELU":
        if value < 0:
            value *= 0.1

    return value


def activation_derivative(value,output):
    if output:
        return 1
    if activation_fuction == "RELU":
        if value > 0:
            return 1
        else:
            return 0

    elif activation_fuction == "LEAKY RELU":
        if value > 0:
            return 1
        else:
            return 0.1

class Neuron:
    def __init__(self, inputs):
        self.inputs = inputs #Number of Inputs
        self.b = 0  #Bias of the neuron
        self.w = [] # Array of weights
        self.last_input = [] #Stores last input for back propogation
        self.last_tot = 0 #Stores last output for back propogation
        self.delta = 0 # error
        self.output = False

        for i in range(self.inputs):
            self.w.append(0)

    def forward(self,inputs):
        self.last_input = inputs
        tot = 0
        for i in range(self.inputs):
            tot += inputs[i] * self.w[i]
        tot += self.b
        self.last_tot = tot
        tot = activation(tot,self.output)
        self.last_output = tot
        return tot


def layer(inputs,neurons):
    tlayer = []
    for i in range(neurons):
        tlayer.append(Neuron(inputs))
    if neurons ==1:
        tlayer[0].output = True
    return tlayer

class nn:
    def __init__(self,layers, sizes):
        self.network = []
        for i in range(layers):
            current_layer = layer(sizes[i],sizes[i+1])
            self.network.append(current_layer)
    def initialise(self):
        for current_layer in self.network:
            for neuron in current_layer:
                # Small Xavier weight scaling to keep initial forward passes stable
                limit = math.sqrt(6 / (neuron.inputs + len(current_layer)))
                for i in range(neuron.inputs):
                    neuron.w[i] = random.uniform(-limit, limit)
                neuron.b = 0
    def forward(self, x):
        for current_layer in self.network:
            outputs = []
            for neuron in current_layer:
                outputs.append(neuron.forward(x))
            x = outputs
        return x

    def backprop_output(self, target):
        output_layer = self.network[-1]
        output_neuron = output_layer[0]

        prediction = output_neuron.last_output

        output_neuron.delta = (
            prediction - target
        ) * activation_derivative(output_neuron.last_tot,output_neuron.output)


    def backprop_hidden(self):
        for l in range(len(self.network) - 2, -1, -1):

            current_layer = self.network[l]
            next_layer = self.network[l + 1]

            for i, neuron in enumerate(current_layer):

                error = 0

                for next_neuron in next_layer:
                    error += next_neuron.w[i] * next_neuron.delta

                neuron.delta = error * activation_derivative(neuron.last_tot,neuron.output)
    def update_weights(self, learning_rate):
        for current_layer in self.network:
            for neuron in current_layer:

                for i in range(neuron.inputs):
                    gradient = neuron.delta * neuron.last_input[i]
                    neuron.w[i] -= learning_rate * gradient

                neuron.b -= learning_rate * neuron.delta
# --- Data Preparation ---
print("Loading California Housing Dataset...")
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split Train (80%) and Test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize inputs (Zero mean, unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Initialize Network ---
# 8 input features -> 12 -> 8 -> 4 -> 1 output prediction
network = nn(5, [8,12,8,8,4,1])
network.initialise()


def train_step(x, target):
    prediction = network.forward(x)
    loss = 0.5 * (prediction[0] - target) ** 2
    network.backprop_output(target)
    network.backprop_hidden()
    network.update_weights(learning_rate)
    return loss


# --- Training Loop ---
print("Training Network...")
epochs = 25

for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X_train)):
        total_loss += train_step(X_train[i], y_train[i])

    avg_loss = total_loss / len(X_train)
    print(
        f"Epoch {epoch + 1}/{epochs} - Mean Squared Error Loss: {avg_loss:.4f}"
    )

print("\nTraining Complete.")

# --- Evaluation on Test Set ---
print("\nTesting Network on Unseen Samples...")
for i in range(10):
    pred = network.forward(X_test[i])[0]
    actual = y_test[i]
    # Fixed number syntax inside print statement (100000 instead of 100,000)
    print(
        f"Sample {i+1} -> Predicted: ${pred*100000:,.0f} | Actual: ${actual*100000:,.0f}"
    )
    print(X_test[i])
