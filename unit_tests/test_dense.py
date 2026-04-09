from modules.dense import Dense
import numpy as np

def test_dense_forward_large():
    np.random.seed(42)

    batch_size = 4
    input_dim = 128
    output_dim = 64
    
    for dense_algo in [0, 1, 2]:
        
        # Create dummy input
        input_data = np.random.randn(batch_size, input_dim).astype(np.float32)

        # Initialize Dense layer
        dense = Dense(in_features=input_dim, out_features=output_dim, dense_algo=dense_algo)

        # Set known weights and bias
        weight = np.random.randn(input_dim, output_dim).astype(np.float32)
        bias = np.random.randn(output_dim).astype(np.float32)
        dense.weights = weight
        dense.biases = bias

        # Forward pass
        output = dense.forward(input_data)
        expected_output = np.dot(input_data, weight) + bias

        assert np.allclose(output, expected_output, atol=1e-5), "Dense forward output mismatch (large case)"
        print(f"✅ Dense layer forward (algo={dense_algo}) pass test (large case) passed.")

# Run the test
test_dense_forward_large()
