import numpy as np


# PISTA: es esta la mejor forma de hacer una matmul?
def matmul_biasses(A, B, C, bias):
    m, p, n = A.shape[0], A.shape[1], B.shape[1]

    for i in range(m):
        for j in range(n):
            for k in range(p):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] += bias[j]
    return C


def im2col(x, K, stride=1, H_out=None, W_out=None):
    # Numero de canales, altura y ancho de la imagen
    C, H, W = x.shape

    if H_out is None:
        H_out = H - K + 1
    if W_out is None:
        W_out = W - K + 1

    cols = np.zeros((C * K * K, H_out * W_out), dtype=np.float32)

    for i in range(H_out):
        for j in range(W_out):
            """
            Extrae un parche de tamaño KxK para cada canal y lo aplana en un vector, que se almacena como una columna en la matriz de salida.
            Por ejemplo, si la imagen de entrada tiene 3 canales (C=3) y el tamaño del kernel es 3x3 (K=3), entonces cada parche extraído tendrá 3x3x3=27 elementos. Si la imagen de entrada es de tamaño 32x32 (H=32, W=32) y el stride es 1, entonces habrá 30x30=900 posiciones donde el kernel puede colocarse, lo que resultará en una matriz de salida de tamaño 27x900 (C*K*K x H_out*W_out).
            """
            patch = x[
                :, i * stride : i * stride + K, j * stride : j * stride + K
            ].reshape(-1)
            cols[:, i * W_out + j] = patch

    return cols


def dense_numpy(X, W, b):
    """
    NumPy-optimized dense layer computation using vectorized operations.
    Leverages BLAS backend for fast matrix multiplication.

    Args:
        X: Input matrix (batch_size, in_features)
        W: Weight matrix (in_features, out_features)
        b: Bias vector (out_features,)

    Returns:
        Y: Output matrix (batch_size, out_features)
    """
    return np.dot(X, W) + b


def dense_gemm(X, W, Y, b):
    N, K = X.shape
    M = W.shape[1]

    for n in range(N):
        for k in range(K):
            for m in range(M):
                Y[n][m] += X[n][k] * W[k][m]

    Y += b
    return Y
