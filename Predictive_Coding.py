import numpy as np
from ETL import UseExtract_Excel

class InferenceResult:
    """
    Container for inference results
    """
    def __init__(self, inputState, hiddenState, outputState, inputError, hiddenError, outputError):
        self.inputState = inputState
        self.hiddenState = hiddenState
        self.outputState = outputState
        self.inputError = inputError
        self.hiddenError = hiddenError
        self.outputError = outputError


class PredictiveCodingNetwork:
    def __init__(
        self,
        inputSize,
        hiddenSize,
        outputSize,
        learningRate=0.01,
        inputVariance=1.0,
        hiddenVariance=1.0,
        outputVariance=1.0
    ):
        self.learningRate = learningRate

        # Variance terms (precision = 1 / variance)
        self.inputVariance = inputVariance
        self.hiddenVariance = hiddenVariance
        self.outputVariance = outputVariance

        self.inputPrecision = 1.0 / inputVariance
        self.hiddenPrecision = 1.0 / hiddenVariance
        self.outputPrecision = 1.0 / outputVariance

        # Generative weights
        self.weightsHiddenToInput = np.random.randn(inputSize, hiddenSize) * 0.1
        self.weightsOutputToHidden = np.random.randn(hiddenSize, outputSize) * 0.1

        self.allPredictions = []
        self.allTargets = []

    # -------------------------------------------------
    # Nonlinear activation functions
    # -------------------------------------------------
    def activation(self, x):
        return np.tanh(x)

    def activationDerivative(self, x):
        return 1.0 - np.tanh(x) ** 2

    # -------------------------------------------------
    # Inference
    # -------------------------------------------------
    def infer(self, inputData, targetOutput=None, steps=40):
        inputState = np.array(inputData, dtype=float)

        hiddenState = np.random.randn(self.weightsHiddenToInput.shape[1]) * 0.01
        outputState = np.random.randn(self.weightsOutputToHidden.shape[1]) * 0.01

        for _ in range(steps):

            # Nonlinear latent activities
            hiddenActivity = self.activation(hiddenState)
            outputActivity = self.activation(outputState)

            # Top-down predictions
            predictedInput = self.weightsHiddenToInput @ hiddenActivity
            predictedHidden = self.weightsOutputToHidden @ outputActivity

            # Precision-weighted prediction errors
            inputError = self.inputPrecision * (inputState - predictedInput)
            hiddenError = self.hiddenPrecision * (hiddenState - predictedHidden)

            if targetOutput is not None:
                targetOutput = np.array(targetOutput, dtype=float)
                outputError = self.outputPrecision * (
                    targetOutput - outputActivity
                )
            else:
                outputError = self.outputPrecision * (-outputActivity)

            # Bottom-up feedback
            inputFeedback = (
                self.weightsHiddenToInput.T @ inputError
            ) * self.activationDerivative(hiddenState)

            hiddenFeedback = (
                self.weightsOutputToHidden.T @ hiddenError
            ) * self.activationDerivative(outputState)

            # State updates
            hiddenState += self.learningRate * (
                inputFeedback - hiddenError
            )

            outputState += self.learningRate * (
                hiddenFeedback + outputError
            )

        return InferenceResult(
            inputState,
            hiddenState,
            outputState,
            inputError,
            hiddenError,
            outputError
        )

    # -------------------------------------------------
    # Weight updates
    # -------------------------------------------------
    def updateWeights(
        self,
        hiddenState,
        outputState,
        inputError,
        hiddenError
    ):
        hiddenActivity = self.activation(hiddenState)
        outputActivity = self.activation(outputState)

        self.weightsHiddenToInput += self.learningRate * np.outer(
            inputError,
            hiddenActivity
        )

        self.weightsOutputToHidden += self.learningRate * np.outer(
            hiddenError,
            outputActivity
        )

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------
    def computeRSquared(self):
        predictions = np.array(self.allPredictions)
        targets = np.array(self.allTargets)

        if len(targets) < 2:
            return 0.0

        meanTarget = np.mean(targets)
        totalVariance = np.sum((targets - meanTarget) ** 2)

        if totalVariance == 0:
            return 0.0

        residualVariance = np.sum((targets - predictions) ** 2)

        return 1.0 - (residualVariance / totalVariance)

    # -------------------------------------------------
    # Training
    # -------------------------------------------------
    def train(self, dataset, epochs=100, steps=40):
        for epoch in range(epochs):

            self.allTargets = []
            self.allPredictions = []

            for inputData, targetOutput in dataset:

                results = self.infer(
                    inputData,
                    targetOutput=targetOutput,
                    steps=steps
                )

                hiddenState = results.hiddenState
                outputState = results.outputState
                inputError = results.inputError

                targetOutput = np.array(targetOutput, dtype=float)

                # Predicted output uses nonlinear activity
                predictedOutput = self.activation(outputState)

                self.allTargets.append(targetOutput[0])
                self.allPredictions.append(predictedOutput[0])

                # Hidden error using supervised output
                predictedHidden = self.weightsOutputToHidden @ targetOutput
                hiddenError = self.hiddenPrecision * (
                    hiddenState - predictedHidden
                )

                self.updateWeights(
                    hiddenState,
                    targetOutput,
                    inputError,
                    hiddenError
                )

            r2 = self.computeRSquared()
            print("Epoch", epoch, "| R^2 =", r2)

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------
    def predict(self, inputData, steps=40):
        results = self.infer(inputData, targetOutput=None, steps=steps)
        return self.activation(results.outputState)

    # -------------------------------------------------
    # Save / Load
    # -------------------------------------------------
    def saveModel(self, filename):
        try:
            np.savez(
                filename,
                weightsHiddenToInput=self.weightsHiddenToInput,
                weightsOutputToHidden=self.weightsOutputToHidden
            )
        except Exception as e:
            print("Save failed:", e)

    def loadModel(self, filename):
        try:
            data = np.load(filename)
            self.weightsHiddenToInput = data["weightsHiddenToInput"]
            self.weightsOutputToHidden = data["weightsOutputToHidden"]
        except Exception as e:
            print("Load failed:", e)


# -------------------------------------------------
# Dataset
# -------------------------------------------------
def imperfectLinear():
    return [
        ([0.1, 0.2], [0.28]),
        ([0.2, 0.1], [0.22]),
        ([0.3, 0.4], [0.65]),
        ([0.4, 0.3], [0.60]),
        ([0.5, 0.6], [1.05]),
        ([0.6, 0.5], [1.00]),
    ]


# -------------------------------------------------
# Run
# -------------------------------------------------

def analyse_airfoil_noise_dataset():
    spreadsheet_name = "airfoil_noise"
    workbook_name = "airfoil_noise.xlsx"
    column_names = [
        "frequency",
        "attack angle",
        "chord-length",
        "free-stream-velocity",
        "suction-side-displacement-thickness",
        "scaled-sound-pressure"
    ]
    remove_headings = True
    keep_first_column = False
    sort = False
    standardisation = True
    raw_data = UseExtract_Excel(
        spreadsheet_name,
        workbook_name,
        column_names,
        remove_headings,
        keep_first_column,
        sort,
        standardisation
    )
    # First 5 columns = inputs
    input_columns = raw_data[:-1]
    # Last column = target output
    output_column = raw_data[-1]

    dataset = []
    # Convert column-wise data into row-wise training pairs
    number_of_rows = len(output_column)
    for row in range(number_of_rows):
        input_data = [
            input_columns[0][row],
            input_columns[1][row],
            input_columns[2][row],
            input_columns[3][row],
            input_columns[4][row]
        ]
        output_data = [output_column[row]]
        dataset.append((input_data, output_data))
    return dataset

# def analyse_airfoil_noise_dataset():
#     spreadsheet_name = "airfoil_noise"
#     workbook_name = "airfoil_noise.xlsx"
#     column_names = [
#         "frequency",
#         "attack angle",
#         "chord-length",
#         "free-stream-velocity",
#         "suction-side-displacement-thickness",
#         "scaled-sound-pressure"
#     ]

#     remove_headings = True
#     keep_first_column = False
#     sort = False
#     standardisation = False

#     raw_data = UseExtract_Excel(
#         spreadsheet_name,
#         workbook_name,
#         column_names,
#         remove_headings,
#         keep_first_column,
#         sort,
#         standardisation
#     )

#     # First 5 columns = inputs
#     input_columns = raw_data[:-1]

#     # Last column = target output
#     output_column = raw_data[-1]

#     # Min-max normalise every column to range [-1, 1]
#     for i in range(len(input_columns)):
#         column = input_columns[i]
#         min_val = min(column)
#         max_val = max(column)

#         input_columns[i] = [
#             2 * ((x - min_val) / (max_val - min_val)) - 1
#             for x in column
#         ]

#     min_val = min(output_column)
#     max_val = max(output_column)

#     output_column = [
#         2 * ((x - min_val) / (max_val - min_val)) - 1
#         for x in output_column
#     ]

#     dataset = []

#     # Convert column-wise data into row-wise training pairs
#     number_of_rows = len(output_column)

#     for row in range(number_of_rows):
#         input_data = [
#             input_columns[0][row],
#             input_columns[1][row],
#             input_columns[2][row],
#             input_columns[3][row],
#             input_columns[4][row]
#         ]

#         output_data = [output_column[row]]

#         dataset.append((input_data, output_data))

#     return dataset

def runPredictiveCoding(dataset):

    network = PredictiveCodingNetwork(
        inputSize=5,
        hiddenSize=16,
        outputSize=1,
        learningRate=0.02,
        inputVariance=1.0,
        hiddenVariance=0.5,
        outputVariance=0.2
    )

    network.train(dataset)

    #prediction = network.predict([0.4, 0.3])
    #print("Prediction:", prediction)


if __name__ == "__main__":
    dataset = analyse_airfoil_noise_dataset()
    runPredictiveCoding(dataset)