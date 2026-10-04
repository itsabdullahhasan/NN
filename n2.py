import random

# Network Parameters
global activation_fuction
activation_fuction = "LEAKY RELU"
learning_rate = 0.001
global total_loss


def activation(value):
    if activation_fuction == "RELU":
        if value < 0:
            value = 0

    elif activation_fuction == "LEAKY RELU":
        if value < 0:
            value *= 0.1

    return value


def activation_derivative(value):
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


        for i in range(self.inputs):
            self.w.append(0)

    def forward(self,inputs):
        self.last_input = inputs
        tot = 0
        for i in range(self.inputs):
            tot += inputs[i] * self.w[i]
        tot += self.b
        self.last_tot = tot
        tot = activation(tot)
        self.last_output = tot
        return tot


def layer(inputs,neurons):
    tlayer = []
    for i in range(neurons):
        tlayer.append(Neuron(inputs))
    return tlayer

class nn:
    def __init__(self,layers, sizes):
        self.network = []
        for i in range(layers):
            current_layer = layer(sizes[i],sizes[i+1])
            self.network.append(current_layer)
    def initialise(self):
        for layer in self.network:
            for neuron in layer:
                for i in range(neuron.inputs):
                    weight = random.uniform(-0.1,0.1)
                    neuron.w[i] = weight
                neuron.b = random.uniform(-0.1,0.1)
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
        ) * activation_derivative(output_neuron.last_tot)


    def backprop_hidden(self):
        for l in range(len(self.network) - 2, -1, -1):

            current_layer = self.network[l]
            next_layer = self.network[l + 1]

            for i, neuron in enumerate(current_layer):

                error = 0

                for next_neuron in next_layer:
                    error += next_neuron.w[i] * next_neuron.delta

                neuron.delta = error * activation_derivative(neuron.last_tot)
    def update_weights(self, learning_rate):
        for current_layer in self.network:
            for neuron in current_layer:

                for i in range(neuron.inputs):
                    gradient = neuron.delta * neuron.last_input[i]
                    neuron.w[i] -= learning_rate * gradient

                neuron.b -= learning_rate * neuron.delta
#Main code
network = nn(4, [1, 12, 8, 4, 1])
print("Created Network")
network.initialise()
print("Initialsied Network")

def createDataSet():
    dataSet = []
    for i in range(-500,500):
        x = i / 100
        y = x **2
        dataSet.append([x,y])
    return dataSet

def train(x,y):
    prediction = network.forward([x])
    loss = 0.5 * (prediction[0] - data[1]) ** 2
    network.backprop_output(y)
    network.backprop_hidden()
    network.update_weights(learning_rate)
    return loss

dataset = createDataSet()
print("Created dataset")

print("Training...")
total_loss = 0
for epoch in range(250):
    prevloss = total_loss
    total_loss = 0  
    for data in dataset:
        total_loss += train(data[0],data[1])
    if epoch % 10 == 0:
        print("Epoch:", epoch, "Loss:", total_loss)
        print("Reduced")
print("Training Complete")

"""
for l, current_layer in enumerate(network.network):
    print("Layer", l)

    for i, neuron in enumerate(current_layer):
        print(
            "Neuron", i,
            "weights:", neuron.w,
            "bias:", neuron.b
        )
"""

print("Test")

for i in range(1000):
    x = float(input())

    prediction = network.forward([x])

    print("Prediction:", prediction[0])
    print("Expected:", x **2)
    print("Error:", (prediction[0] - (x**2)) **2)

print("Testing Complete")
