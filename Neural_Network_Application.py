import math
import random

class ReLuFunction(): #Implementation of ReLU activation function
    @staticmethod
    def compute_pre_activation(weight, bias, value):
        return weight * value + bias

    @staticmethod
    def compute_activation(z):
        #Standard ReLu
        return max(0, z)

class DerivativeReLu(): #Implementation of the derivative of a ReLU activation function
    @staticmethod
    def compute(value):
        if value > 0:
            return 1
        return 0

class Sigmoid:
    @staticmethod
    def compute_activation(z):
        # Clamp z to a safe range to prevent exp overflow, then use numerically-stable formulas.
        # 500 is safe for double precision: exp(500) is large but handled via the stable branch.
        if z > 500:
            z = 500.0
        elif z < -500:
            z = -500.0

        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

class DerivativeSigmoid:
    @staticmethod
    def compute(a):
        return a * (1 - a)

class Neuron:
    def __init__(self, weights, bias, activation):
        self.weights = weights  # list
        self.bias = bias
        self.activation = activation
        self.z = None
        self.a = None

    def compute(self, inputs):
        # inputs is a list (hidden_outputs or input features)
        self.z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        self.a = self.activation.compute_activation(self.z)
        return self.a

    def adjust_weight(self, lr, grad_hidden_weight):
        # not used in current code, but fixed attribute name
        # grad_hidden_weight expected to be a list if neuron has multiple weights
        if isinstance(grad_hidden_weight, list):
            for i, gw in enumerate(grad_hidden_weight):
                self.weights[i] -= lr * gw
        else:
            # single weight case (legacy)
            self.weights[0] -= lr * grad_hidden_weight

    def adjust_bias(self, lr, grad_hidden_bias):
        self.bias -= lr * grad_hidden_bias

class NeuralNetwork():
    def __init__(self, x, y, learning_rate, size_hidden=20, size_output=1):
        # changed parameter order to (size_hidden, size_output) to match run_test usage
        self.x = x
        self.y = y
        self.size = size_hidden 
        self.size_output = size_output
        self.weight= 0.5
        self.bias = 0.4
        self.n = min(len(x), len(y))
        self.derivative_activation = DerivativeReLu()
        self.activation =  ReLuFunction()
        self.learning_rate = learning_rate

        # determine input dimensionality from the first sample (defaults to 1)
        if self.n > 0 and isinstance(self.x[0], (list, tuple)):
            self.input_dim = len(self.x[0])
        else:
            self.input_dim = 1

        self.count = 0

    def generate_hidden_layer(self):
        # create hidden neurons with weights sized to the input dimensionality
        # use smaller initial weights to reduce risk of exploding activations
        self.hidden_layer = []
        for i in range(self.size):
            self.hidden_layer.append(
                Neuron(
                    weights=[random.uniform(-0.1, 0.1) for _ in range(self.input_dim)],
                    bias=random.uniform(-0.05, 0.05),
                    activation=ReLuFunction()
                )
            )

    def generate_output_layer(self):
        self.output_layer = []
        for i in range(self.size_output):
            self.output_layer.append(
                Neuron(
                    weights = [random.uniform(-0.1, 0.1) for _ in range(self.size)],
                    bias = random.uniform(-0.05, 0.05),
                    activation = Sigmoid()
                )
            )

    def initialise_neurons(self):
        self.generate_hidden_layer()
        self.generate_output_layer()

    def loss(self, predictions, targets):
        """
        predictions: list[float]
        targets: list[float]
        """
        return sum((p - y) ** 2 for p, y in zip(predictions, targets)) / len(targets)

    def forward_hidden(self, x):
        hidden_outputs = []
        for neuron in self.hidden_layer:
            # compute() already stores neuron.z and neuron.a
            hidden_outputs.append(neuron.compute(x))
        return hidden_outputs

    def forward_output(self, hidden_outputs):
        outputs = []
        for neuron in self.output_layer:
            outputs.append(neuron.compute(hidden_outputs))
        return outputs

    def forward_activation(self):
        # activation
        activation = math.tanh(self.z_backprop)
        return activation

    def compute_rates(self, prediction, y_target):
        # dL/da (MSE)
        self.loss_rate = 2 * (prediction - y_target)
        # derivative of tanh(z)
        self.activation_rate = 1

    def compute_output_deltas(self, predictions, targets):
        deltas = []
        for p, y in zip(predictions, targets):
            sigmoid_derivative = p * (1 - p)
            deltas.append(2 * (p - y) * sigmoid_derivative)
        return deltas

    def compute_hidden_deltas(self, output_deltas):
        hidden_deltas = []
        for j, neuron in enumerate(self.hidden_layer):
            error = 0
            for k, delta in enumerate(output_deltas):
                error += delta * self.output_layer[k].weights[j]
            da_dz = self.derivative_activation.compute(neuron.z)
            hidden_deltas.append(error * da_dz)
        return hidden_deltas

    def compute_output_gradients(self, hidden_outputs, delta_output):
        """
        hidden_outputs: list[float]       length = size_hidden
        delta_output:   list[float]       length = size_output
        """
        self.grad_output_weights = []   # shape: [size_output][size_hidden]
        self.grad_output_biases = []      # shape: [size_output]
        for o, delta_o in enumerate(delta_output):
            # weights for output neuron o
            grad_w_o = []
            for h in hidden_outputs:
                grad_w_o.append(delta_o * h)
            # append inside the loop (fixes previous indentation bug)
            self.grad_output_weights.append(grad_w_o)
            self.grad_output_biases.append(delta_o)

    def compute_hidden_gradients(self, x, hidden_deltas):
        # ensure x is a flat feature list
        if not isinstance(x, (list, tuple)):
            x_features = [x]
        else:
            x_features = list(x)

        # If input_dim and sample length mismatch, try to adjust:
        if len(x_features) != self.input_dim:
            # If sample has fewer features than expected, pad with zeros.
            if len(x_features) < self.input_dim:
                x_features = x_features + [0.0] * (self.input_dim - len(x_features))
            else:
                # If sample has more features than expected, truncate.
                x_features = x_features[:self.input_dim]

        self.grad_hidden_weights = []
        self.grad_hidden_biases = []
        for delta in hidden_deltas:
            self.grad_hidden_weights.append([delta * xi for xi in x_features])
            self.grad_hidden_biases.append(delta)

    def update_weights_and_biases(self):
        lr = self.learning_rate
        max_grad = 1.0   # gradient clipping threshold

        # verify gradients exist
        if not hasattr(self, "grad_output_weights") or not hasattr(self, "grad_output_biases"):
            raise RuntimeError("Output gradients missing. Call compute_output_gradients before update.")
        if not hasattr(self, "grad_hidden_weights") or not hasattr(self, "grad_hidden_biases"):
            raise RuntimeError("Hidden gradients missing. Call compute_hidden_gradients before update.")

        # Validate shapes to provide a clear error if there's a mismatch between network
        # configuration (size_output / input dim) and targets / gradients.
        if len(self.grad_output_weights) != len(self.output_layer) or len(self.grad_output_biases) != len(self.output_layer):
            raise ValueError(
                f"Gradient/output-layer size mismatch: len(grad_output_weights)={len(self.grad_output_weights)}, "
                f"len(grad_output_biases)={len(self.grad_output_biases)}, len(output_layer)={len(self.output_layer)}. "
                "Ensure your target vectors have the same length as size_output."
            )

        # update output-layer weights and biases with gradient clipping
        for k, neuron in enumerate(self.output_layer):
            grad_w = self.grad_output_weights[k]
            # protect against mismatched hidden-output dimension
            if len(grad_w) != len(neuron.weights):
                raise ValueError(
                    f"Gradient/weight size mismatch for output neuron {k}: len(grad_w)={len(grad_w)}, len(neuron.weights)={len(neuron.weights)}"
                )
            for j in range(len(neuron.weights)):
                g = grad_w[j]
                # clip gradient
                if g > max_grad:
                    g = max_grad
                elif g < -max_grad:
                    g = -max_grad
                neuron.weights[j] -= lr * g
            gb = self.grad_output_biases[k]
            if gb > max_grad:
                gb = max_grad
            elif gb < -max_grad:
                gb = -max_grad
            neuron.bias -= lr * gb

        # validate hidden gradient shapes
        if len(self.grad_hidden_weights) != len(self.hidden_layer) or len(self.grad_hidden_biases) != len(self.hidden_layer):
            raise ValueError(
                f"Gradient/hidden-layer size mismatch: len(grad_hidden_weights)={len(self.grad_hidden_weights)}, "
                f"len(grad_hidden_biases)={len(self.grad_hidden_biases)}, len(hidden_layer)={len(self.hidden_layer)}"
            )

        # update hidden-layer weights and biases with gradient clipping
        for i, neuron in enumerate(self.hidden_layer):
            grad_w_h = self.grad_hidden_weights[i]
            if len(grad_w_h) != len(neuron.weights):
                raise ValueError(
                    f"Gradient/weight size mismatch for hidden neuron {i}: len(grad_w_h)={len(grad_w_h)}, len(neuron.weights)={len(neuron.weights)}"
                )
            for j in range(len(neuron.weights)):
                g = grad_w_h[j]
                # clip gradient
                if g > max_grad:
                    g = max_grad
                elif g < -max_grad:
                    g = -max_grad
                neuron.weights[j] -= lr * g
            gb_h = self.grad_hidden_biases[i]
            if gb_h > max_grad:
                gb_h = max_grad
            elif gb_h < -max_grad:
                gb_h = -max_grad
            neuron.bias -= lr * gb_h

            # quick safety: ensure weights/biases remain finite
            for w in neuron.weights:
                if not math.isfinite(w):
                    raise RuntimeError("Non-finite weight detected in hidden layer during update.")
            if not math.isfinite(neuron.bias):
                raise RuntimeError("Non-finite bias detected in hidden layer during update.")

        # safety check for output layer
        for neuron in self.output_layer:
            for w in neuron.weights:
                if not math.isfinite(w):
                    raise RuntimeError("Non-finite weight detected in output layer during update.")
            if not math.isfinite(neuron.bias):
                raise RuntimeError("Non-finite bias detected in output layer during update.")

    def predict(self, value):
        hidden_outputs = self.forward_hidden(value)
        prediction_value = self.forward_output(hidden_outputs)
        return prediction_value

    def classify(self, prediction_value, outcomes):
        boundary = 0.5 #0.5 is the boundary between outcomes
        # prediction_value may be a list; handle single-output case
        val = prediction_value[0] if isinstance(prediction_value, list) else prediction_value
        if val > boundary:
            return outcomes[1]
        return outcomes[0]

    def r_squared(self, y_true, y_pred):
        """
        y_true: list of real values
        y_pred: list of predicted values
        """
        mean_y = sum(y_true) / len(y_true)
        ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
        ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
        if ss_tot == 0:
            return 0
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def train(self, epochs, verbose=False):
        print("TRAINING...")

        for epoch in range(epochs):
            total_loss = 0.0

            for i in range(self.n):
                x = self.x[i]              # vector input OR scalar
                y_target = [self.y[i]]
                # MUST be a list (vector)
                # ---- FORWARD ----
                hidden_outputs = self.forward_hidden(x)
                predictions = self.forward_output(hidden_outputs)  # list
                # ---- LOSS ----
                loss = self.loss(predictions, y_target)
                total_loss += loss
                # ---- BACKPROP ----
                output_deltas = self.compute_output_deltas(predictions, y_target)
                hidden_deltas = self.compute_hidden_deltas(output_deltas)
                self.compute_output_gradients(hidden_outputs, output_deltas)
                self.compute_hidden_gradients(x, hidden_deltas)
                try:
                    self.update_weights_and_biases()
                except RuntimeError as e:
                    print("Training stopped: numeric instability detected:", e)
                    return []  # abort training, return empty results to avoid printing NaNs

            if verbose and epoch % 1000 == 0:
                print(f"Epoch {epoch:5d} | Loss = {total_loss / self.n:.6f}")
        print("")
        # ---- FINAL TEST OUTPUT ----
        results = []
        y_true_list = []
        y_pred_list = []
        for x_val, y_true in zip(self.x, self.y):
            y_pred = self.predict(x_val)
            results.append((x_val, y_true, y_pred))

            y_true_list.append(y_true)
            y_pred_list.append(y_pred[0])

        r2 = self.r_squared(y_true_list, y_pred_list)
        print(f"R2 score: {r2}")
        try:
            print(f"Correlation Coefficient (R): {math.sqrt(r2)}")
        except:
            print("Math exception occurred: Variance is too small.")

        return results
 
    def train_and_observe_results(self, epochs, verbose):
        #This assumes the Neural Network has data
        self.initialise_neurons()
        results = self.train(epochs, verbose)
        #print("==== RESULTS ====")
        count = 1
        avg_sum = 0 
        for x_val, y_true, y_pred in results:
            avg_sum += (y_pred[0] - y_true)
            count += 1
            #print(f"x={x_val} | y_true={y_true} | y_pred={y_pred} ")
        print(f"Average discrepancy between predictions and outcomes: {avg_sum / (count - 1)} ")
        print("") #Line break