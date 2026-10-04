
global speed 
speed = 0.0005
def relu(x):
    if x < 0:
        return 0
    return x

class neuron:
    def __init__(self,hidden,number):
        self.weight = 0
        self.bis = 0
        self.hidden = hidden
        if hidden == False:
            self.weight2 = 0
            self.weight3 = 0
        else:
            self.number = number
    def pred(self,input):
        if self.hidden:
            pred = relu((input * self.weight) + self.bis)
        else:
            pred = input[0] * self.weight + input[1] * self.weight2 + input[2] * self.weight3 + self.bis
        return pred
    def trin(self,input,output):
        prediction = self.pred(input)
        if self.hidden == False:
            error = prediction - output
            loss = error * error
            g1 = 2 * error * input[0]
            g2 = 2 * error * input[1]
            g3 = 2 * error * input[2]
            gb = 2 * error
            self.vlues = [g1,g2,g3,gb]
        else:
            z = input * self.weight + self.bis
            if z >= 0:
                rlg = 1
            else:
                rlg = 0
            if self.number == 1:
                w = out.weight
            elif self.number == 2:
                w = out.weight2
            else:
                w = out.weight3
            error = output
            grdient = 2 * error * w * rlg 
            self.vlues = [grdient,input]
            
    def trin2(self):
        if self.hidden == False:
            g = self.vlues
            self.weight -= speed * g[0]
            self.weight2 -= speed * g[1]
            self.weight3 -= speed * g[2]
            self.bis -= speed * g[3]
        else:
            self.weight -= self.vlues[0] * speed * self.vlues[1]
            self.bis -= self.vlues[0] * speed
            
            
            

            
            

dtset = []
for i in range(0,501):
    i = i / 100
    b = i*i
    c = [i,b]
    dtset.append(c)

n1 = neuron(True,1)
n2 = neuron(True,2)
n3 = neuron(True,3)
out = neuron(False,0)

n1.weight = 0.4
n1.bis = 0.2

n2.weight = -0.3
n2.bis = -0.1

n3.weight = 0.2
n3.bis = 0.1

out.weight = 0.3
out.weight2 = 0.4
out.weight3 = -0.2
out.bis = 0.1

for epoch in range(2000):
    tloss = 0
    for i in dtset:
        x = i[0]
        y = i[1]
        h1 = n1.pred(x)
        h2 = n2.pred(x)
        h3 = n3.pred(x)
        hiddenOutput = [h1,h2,h3]
        pred = out.pred(hiddenOutput)
        error = pred - y
        tloss += (error * error)
        
        out.trin(hiddenOutput, y)
        n1.trin(x, error)
        n2.trin(x, error)
        n3.trin(x, error)
        out.trin2()
        n1.trin2()
        n2.trin2()
        n3.trin2()
        
    if epoch % 100 == 0:
        print("epoch:", epoch, "loss:", tloss / 500)
print("trined")
print("w1:", out.weight)
print("w2:", out.weight2)
print("w3:", out.weight3)
print("bias:", out.bis)
while True:
    newx = float(input())
    if newx == 0:
        break
    h1 = n1.pred(newx)
    h2 = n2.pred(newx)
    h3 = n3.pred(newx)
    hiddenOutput = [h1,h2,h3]
    pred = out.pred(hiddenOutput)
    print(round(pred,5))
