import numpy as np
learningRate = 0.001
class Layer:
  def __init__(self, inputs, neurons):
    self.weights = 0.01 * np.random.randn(inputs, neurons)
    self.biases = np.zeros((1, neurons))
    self.hidden = True
  def forward(self, inputs):
    self.inputs = inputs
    output = np.dot(inputs, self.weights) + self.biases
    self.z = output
    if self.hidden:
      self.output = np.maximum(0, self.z)
    else:
      self.output = self.z
    return self.output
  def backward(self, dvalues):
    dvalues = dvalues.copy()

    if self.hidden:
        dvalues[self.z <= 0] = 0

    self.dweights = np.dot(self.inputs.T, dvalues)
    self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
    self.dinputs = np.dot(dvalues, self.weights.T)
    self.weights -= learningRate * self.dweights
    self.biases -= learningRate * self.dbiases
    return self.dinputs
      
class Network:
  def __init__(self,structure):
    self.network = []
    for i in range(len(structure) - 1):
      layer = Layer(structure[i],structure[i+1])
      if i == len(structure) - 2:
        layer.hidden = False
      self.network.append(layer)
  def forward(self,inputs):
    outputs = inputs
    for i in self.network:
      outputs = i.forward(outputs)
    return outputs
  
  def backprop(self, output, intended):
    l = output - intended
    dvalues = l
    for layer in reversed(self.network):
        dvalues = layer.backward(dvalues)
    return l**2 / 2
  
nn = Network([1, 4, 3, 1])

data = []
for i in range(-100,100):
  x = i / 100
  y = x* 5
  data.append([x,y])

for epoch in range(1000):
    mse = 0

    for i in data:
        output = nn.forward(np.array([[i[0]]]))
        mse += nn.backprop(output, i[1])

    mse /= len(data)

    if epoch % 100 == 0:
        print(epoch, mse)

while True:
   value = float(input())
   output = nn.forward(np.array([[value]]))
   expected = value * 5
   loss = (output - expected) ** 2 / 2 
   print(f"Output {output} \t Expected {expected} \t Loss {loss}")
