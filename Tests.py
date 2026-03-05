import math 
import random 

class Tests():
    
    def __init__(self, NeuralNetwork):
        self.NeuralNetwork = NeuralNetwork

    def sanity_check():
        x = []
        y = []
        for i in range(-10, 11):
            x.append([i / 10])
            y.append(i/10)

        test_data = {"x":x, "y":y}
        return test_data

    @staticmethod
    def binary_classification():
        x = [[-1], [-0.5], [0], [0.5], [1]]
        y = [0, 0, 0, 1, 1]
        test_data = {"x":x, "y":y}
        return test_data

    @staticmethod
    def xor_test():
        #Tests hidden layer
        x = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
            ]
        y = [0, 1, 1, 0]
        test_data = {"x":x, "y":y}
        return test_data

    @staticmethod
    def noisy_regression():
        # realistic: many samples, single feature
        x = []
        y = []
        for i in range(-20, 21):
            xi = i / 10
            x.append([xi])
            y.append(math.sin(xi) + random.uniform(-0.05, 0.05))
        test_data = {"x":x, "y":y}
        return test_data

    def run_test(self, test_data, epochs):
        x = test_data.get("x")
        y = test_data.get("y")

        size_hidden = 20
        size_output = 1 #1 output node approximates XOR functions better
        learning_rate = 0.005

        neural_network = self.NeuralNetwork(x, y, learning_rate, size_hidden, size_output)
        neural_network.initialise_neurons()

        results = neural_network.train(epochs, verbose=True)

        print("\n=== TEST RESULTS ===")
        for x_val, y_true, y_pred in results:
            print(f"x={x_val} | y_true={y_true} | y_pred={y_pred}")

    def run_all_tests(self):
        epochs = 10000
        print("Sanity Check:")
        test_data = Tests.sanity_check()
        self.run_test(test_data, epochs)
        print("Binary Classification:")
        test_data = Tests.binary_classification()
        self.run_test(test_data, epochs)
        print("XOR Test:")
        test_data = Tests.xor_test()
        self.run_test(test_data, epochs)
        print("Noisy Regression:")
        test_data = Tests.noisy_regression()
        self.run_test(test_data, epochs)
