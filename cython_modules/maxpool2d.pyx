"""
Cython-optimized MaxPool2D implementation.
Provides significant performance improvements over pure Python loops.
"""
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport INFINITY

#Define typed arrays for better performance
ctypedef np.float32_t DTYPE_f32
ctypedef np.int32_t DTYPE_i32

def maxpool_forward_cython(
    np.ndarray[DTYPE_f32, ndim=4] input_array,
    int kernel_size,
    int stride
):
    """
    Cython-optimized MaxPool2D forward pass.
    
    Args:
        input_array: Input tensor of shape (B, C, H, W)
        kernel_size: Size of pooling kernel (square)
        stride: Stride of pooling operation
    
    Returns:
        output: Output tensor of shape (B, C, out_h, out_w)
    """
    cdef int B, C, H, W, KH, KW, SH, SW
    cdef int out_h, out_w
    cdef int b, c, i, j, h, w
    cdef int h_start, h_end, w_start, w_end
    cdef float max_val, val

#Get input dimensions
    B, C, H, W = input_array.shape[0], input_array.shape[1], input_array.shape[2], input_array.shape[3]
    KH, KW = kernel_size, kernel_size
    SH, SW = stride, stride

#Calculate output dimensions
    out_h = (H - KH) // SH + 1
    out_w = (W - KW) // SW + 1

#Allocate output array
    cdef np.ndarray[DTYPE_f32, ndim=4] output = np.zeros((B, C, out_h, out_w), dtype=np.float32)

#Main computation loops
    for b in range(B):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * SH
                    h_end = h_start + KH
                    w_start = j * SW
                    w_end = w_start + KW

#Find maximum in window
                    max_val = -INFINITY
                    
                    for h in range(h_start, h_end):
                        for w in range(w_start, w_end):
                            val = input_array[b, c, h, w]
                            if val > max_val:
                                max_val = val
                    
                    output[b, c, i, j] = max_val
    
    return output
