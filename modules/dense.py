from modules.layer import Layer
from modules.utils import matmul_biasses, dense_gemm, dense_numpy
import numpy as np


class Dense(Layer):
    def __init__(self, in_features, out_features, dense_algo=0, weight_init="he"):
        self.in_features = in_features
        self.out_features = out_features

        if dense_algo == 0:
            self.mode = "numpy"
        elif dense_algo == 1:
            self.mode = "direct"
        elif dense_algo == 2:
            self.mode = "gemm"
        else:
            print(f"Algorithm {dense_algo} not supported, defaulting to numpy")
            self.mode = "numpy"

        if weight_init == "he":
            std = np.sqrt(2.0 / in_features)
            self.weights = (
                np.random.randn(in_features, out_features).astype(np.float32) * std
            )
        elif weight_init == "xavier":
            std = np.sqrt(2.0 / (in_features + out_features))
            self.weights = (
                np.random.randn(in_features, out_features).astype(np.float32) * std
            )
        elif weight_init == "custom":
            self.weights = np.zeros((in_features, out_features), dtype=np.float32)
        else:
            self.weights = np.random.randn(in_features, out_features).astype(
                np.float32
            ) * (1 / in_features**0.5)

        self.biases = np.zeros(out_features, dtype=np.float32)

        self.input = None

    def forward(self, input, training=True):  # input: [batch_size x in_features]
        if self.mode == "numpy":
            return self._forward_numpy(input)
        elif self.mode == "direct":
            return self._forward_direct(input)
        elif self.mode == "gemm":
            return self.__forward_gemm(input)
        else:
            print(f"Mode {self.mode} not supported, defaulting to numpy")
            return self._forward_numpy(input)

    def backward(self, grad_output, learning_rate):
        grad_output = np.array(grad_output).astype(
            np.float32
        )  # Ensure grad_output is float for numerical stability
        batch_size = grad_output.shape[0]

        # Gradient w.r.t. weights
        grad_weights = np.zeros((self.in_features, self.out_features), dtype=np.float32)
        for i in range(self.in_features):
            for j in range(self.out_features):
                for b in range(batch_size):
                    grad_weights[i][j] += self.input[b][i] * grad_output[b][j]
        # Gradient w.r.t. biases
        grad_biases = np.sum(grad_output, axis=0)

        # Gradient w.r.t. input
        grad_input = np.zeros((batch_size, self.in_features), dtype=np.float32)
        for b in range(batch_size):
            for i in range(self.in_features):
                for j in range(self.out_features):
                    grad_input[b][i] += grad_output[b][j] * self.weights[i][j]

        # Update weights and biases
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input

    def _forward_numpy(self, input):
        self.input = np.array(input).astype(
            np.float32
        )  # Ensure input is float for numerical stability

        output = dense_numpy(self.input, self.weights, self.biases).astype(np.float32)
        self.output = output
        return output

    def _forward_direct(self, input):
        self.input = np.array(input).astype(
            np.float32
        )  # Ensure input is float for numerical stability
        batch_size = self.input.shape[0]

        output = np.zeros((batch_size, self.out_features), dtype=np.float32)

        output = matmul_biasses(self.input, self.weights, output, self.biases)
        self.output = output
        return output

    def __forward_gemm(self, input):
        self.input = np.array(input).astype(
            np.float32
        )  # Ensure input is float for numerical stability
        batch_size = self.input.shape[0]

        output = np.zeros((batch_size, self.out_features), dtype=np.float32)
        for b in range(batch_size):
            output[b] = dense_gemm(
                self.input[b : b + 1], self.weights, output[b : b + 1], self.biases
            )

        self.output = output
        return output

    def get_weights(self):
        return {"weights": self.weights, "biases": self.biases}

    def set_weights(self, weights):
        self.weights = weights["weights"]
        self.biases = weights["biases"]
