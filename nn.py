import math

def tanh(x):
    return math.tanh(x)

global speed 
speed = 0.0001
neurons = 20
epochs = 1500
RELU = True
leky = True
def relu(x):
    if x < 0:
        return 0.01 * x
    return x

class outputNeuron:
    def __init__(self,hidden):
        self.b = 0
        self.hidden = hidden
        self.w = []
        for i in range(self.hidden):
            self.w.append(0)
    def pred(self,inputs):  
        tot = 0
        for i in range(self.hidden):
            tot += inputs[i] * self.w[i]
        pred = tot + self.b
        return pred
    def trin(self, inputs, error):
        for i in range(self.hidden):
            self.w[i] -= speed * 2 * error * inputs[i]
        self.b -= speed * 2 * error

#newWeight -= speed *  2 * error * input[0]
#newbis -= speed * 2 * error

class hiddenNeuron:
    def __init__(self,id):
        self.w = 0
        self.b = 0
        self.id = id
    def pred(self, input):
        z = self.w * input + self.b
        if RELU:
            if z > 0:
                self.activation = z
                self.derivative = 1
            else:
                if leky:
                    self.activation = z * 0.01
                    self.derivative = 0.01
                else:
                    self.activation = 0
                    self.derivative = 0
        else:
            self.activation = math.tanh(z)
            self.derivative = 1 - self.activation ** 2

        self.w2 = output.w[self.id]

        return self.activation
    def trin(self,error,input):
        grd = 2 * error * self.derivative * self.w2
        self.w -= grd * speed * input
        self.b -= grd * speed


#new weight -= grdient * speed * input
#grdient = 2 * error * outputweight * rlg
#new bis -= grdient * speed       



def pred(input,idel):
    hiddenOutput = []
    for n in hiddenNeurons:
        hidden = n.pred(input)
        hiddenOutput.append(hidden)
    pred = output.pred(hiddenOutput)
    error = pred - idel
    loss = error * error
    for n in hiddenNeurons:
        n.trin(error,input)
    output.trin(hiddenOutput,error)
    return loss
    

output = outputNeuron(neurons)



hiddenNeurons = []
for i in range(neurons):
    n = hiddenNeuron(i)
    n.w = (i - neurons / 2) * 0.2
    n.b = (i - neurons / 2) * 0.1
    hiddenNeurons.append(n)
    output.w[i] = 0.1


#Crete dtset
dtset = []
for i in range(-1000,1000):
    i = i / 100
    x_norm = i / 5
    y_norm = (i * i) / 25
    c = [x_norm,y_norm]
    dtset.append(c)

#trin
for epoch in range(epochs):
    tloss = 0
    for i in dtset:
        x = i[0]
        y = i[1]
        loss = pred(x,y)
        tloss += loss
        
    if epoch % 50 == 0:
        print("epoch:", epoch, "loss:", tloss / len(dtset))

#Test
print("trining complete")
while True:
    newx = float(input("Enter number: | "))
    if newx == 999:
        break
    newx = newx / 5
    hiddenOutput = []
    for n in hiddenNeurons:
        hidden = n.pred(newx)
        hiddenOutput.append(hidden)
    pred = output.pred(hiddenOutput) * 25
    true = newx * newx * 25
    loss = (pred - true) ** 2
    print(pred, "\t true:", true, "\t loss:", loss)
