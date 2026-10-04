import math

#prmeters
LEARNING_RATE = 0.0005
NUM_NEURONS = 20
EPOCHS = 500
USE_RELU = False
USE_LEAKY = False


class OutputNeuron:
    def __init__(self, num_hidden):
        self.b = 0.0
        self.num_hidden = num_hidden
        self.w = [0.1 for _ in range(num_hidden)]

    def forward(self, hidden_activations):
        tot = sum(h * w for h, w in zip(hidden_activations, self.w))
        return tot + self.b

    def train(self, hidden_activations, error):
        # dL/dw_out = 2 * error * hidden_activation
        for i in range(self.num_hidden):
            self.w[i] -= LEARNING_RATE * 2 * error * hidden_activations[i]
        self.b -= LEARNING_RATE * 2 * error


class HiddenNeuron:
    def __init__(self, neuron_id):
        self.id = neuron_id
        # Small spread initialization
        self.w = (neuron_id - NUM_NEURONS / 2) * 0.1
        self.b = 0.0
        self.activation = 0.0
        self.derivative = 0.0

    def forward(self, x):
        z = self.w * x + self.b
        if USE_RELU:
            if z > 0:
                self.activation = z
                self.derivative = 1.0
            else:
                alpha = 0.01 if USE_LEAKY else 0.0
                self.activation = alpha * z
                self.derivative = alpha
        else:
            self.activation = math.tanh(z)
            self.derivative = 1.0 - math.tanh(z) ** 2

        return self.activation

    def train(self, error, x, output_weight):
        # Gradient = dL/d_pred * d_pred/d_activation * d_activation/dz
        grad = 2 * error * output_weight * self.derivative
        self.w -= LEARNING_RATE * grad * x
        self.b -= LEARNING_RATE * grad


# --- Model Instantiation ---
output_layer = OutputNeuron(NUM_NEURONS)
hidden_layer = [HiddenNeuron(i) for i in range(NUM_NEURONS)]


def step(x_norm, y_target):
    # Forward Pass
    hidden_outputs = [n.forward(x_norm) for n in hidden_layer]
    y_pred = output_layer.forward(hidden_outputs)

    # Compute Loss
    error = y_pred - y_target
    loss = error ** 2

    # Backpropagation (Hidden layer uses output weights before output layer updates)
    for n in hidden_layer:
        n.train(error, x_norm, output_layer.w[n.id])
    output_layer.train(hidden_outputs, error)

    return loss


# --- Dataset Generation (y = x^2 normalized) ---
dataset = []
for i in range(-2000, 2000):
    val = i / 100.0         # -10.0 to 10.0
    x_norm = val / 5.0       # -2.0 to 2.0
    y_norm = (x_norm ** 2)   # 0.0 to 4.0
    dataset.append((x_norm, y_norm))

# --- Training Loop ---
print("Training started...")
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for x_norm, y_norm in dataset:
        total_loss += step(x_norm, y_norm)

    if epoch % 50 == 0 or epoch == 1:
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:4d} | Average Loss: {avg_loss:.6f}")

print("Training complete!\n")

# --- Interactive Testing ---
while True:
    try:
        user_input = input("Enter a number to square (or 999 to quit): | ")
        val = float(user_input)
        if val == 999:
            break

        x_norm = val / 5.0
        hidden_outputs = [n.forward(x_norm) for n in hidden_layer]
        pred_norm = output_layer.forward(hidden_outputs)

        # Scale back to original domain
        pred_actual = pred_norm * 25.0
        true_actual = val ** 2
        diff = abs(pred_actual - true_actual)

        print(f"Predicted: {pred_actual:8.4f} | Actual: {true_actual:8.4f} | Absolute Error: {diff:8.4f}\n")
    except ValueError:
        print("Please enter a valid number.")
