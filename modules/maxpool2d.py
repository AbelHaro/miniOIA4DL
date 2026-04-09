from modules.layer import Layer
import numpy as np

# Try to import Cython-optimized version
try:
    from cython_modules.maxpool2d import maxpool_forward_cython
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride, pool_algo=0):
        """
        MaxPool2D layer with multiple algorithm options.
        """
        
        self.kernel_size = kernel_size
        self.stride = stride
        if pool_algo == 0:
            if not CYTHON_AVAILABLE:
                print("Warning: Cython module not available, falling back to Python")
                self.mode = "direct"
            else:
                self.mode = "cython"
        elif pool_algo == 1:
            self.mode = "direct"
        else:
            print(f"Algorithm {pool_algo} not supported, defaulting to Python")
            self.mode = "direct"
    

    def forward(self, input, training=True):  # input: np.ndarray of shape [B, C, H, W]
        self.input = input
        
        if self.mode == "cython" and CYTHON_AVAILABLE:
            output = maxpool_forward_cython(
                input.astype(np.float32),
                self.kernel_size,
                self.stride
            )
        else:
            output = self._forward_direct(input)
        
        return output

    def backward(self, grad_output, learning_rate=None):
        B, C, H, W = self.input.shape
        grad_input = np.zeros_like(self.input, dtype=grad_output.dtype)
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        r, s = self.max_indices[b, c, i, j]
                        grad_input[b, c, r, s] += grad_output[b, c, i, j]

        return grad_input
    
    def _forward_direct(self, input):
        self.input = input
        B, C, H, W = input.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1

        self.max_indices = np.zeros((B, C, out_h, out_w, 2), dtype=int)
        output = np.zeros((B, C, out_h, out_w),dtype=input.dtype)

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * SH
                        h_end = h_start + KH
                        w_start = j * SW
                        w_end = w_start + KW

                        window = input[b, c, h_start:h_end, w_start:w_end]
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        max_val = window[max_idx]

                        output[b, c, i, j] = max_val
                        self.max_indices[b, c, i, j] = (h_start + max_idx[0], w_start + max_idx[1])

        return output
