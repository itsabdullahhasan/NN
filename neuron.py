class neuron:
    def __init__(self):
        self.weight = 0
        self.bis = 0
        self.loss = 0
    def pred(self,input):
        pred = (input * self.weight) + self.bis
        return pred
    def trin(self, input, output):
        pred = (input * self.weight) + self.bis
        error = pred - output
        loss = error * error
        self.loss = loss
        #djust
        wg = 2 * error * input
        wb = 2 * error
        self.weight = self.weight - (0.01 * wg)
        self.bis = self.bis - (0.01 * wb)
        

dtset = []
for i in range(1000):
    i = i / 100
    b = (i*7) -4
    c = [i,b]
    dtset.append(c)

n1 = neuron()
n2b = 10000000000
n2w = 10000000000

for epoch in range(36):
    tloss = 0
    for i in dtset:
        x = i[0]
        y = i[1]
        n1.trin(x, y)
        tloss += n1.loss
    print(n1.bis, n1.weight)
    print(tloss / 1000)
    if abs(n1.bis - n2b) < 0.0001 and abs(n1.weight - n2w) < 0.0001:
        print("Converged at epoch:", epoch)
        break
    n2b = n1.bis
    n2w = n1.weight

print(round(n1.weight))
print(round(n1.bis))
print("Loss:", tloss / 1000)

newx = float(input())
print(round(n1.pred(newx),1))

