import numpy as np
import sys
import json

def generate_matrix(rows: int, cols: int):
    """
    Generates a matrix of random float values of shape (rows, cols).

    Parameters:
    - rows (int): Number of rows in the matrix.
    - cols (int): Number of columns in the matrix.

    Returns:
    - list[list[float]]: The generated matrix as a nested list.
    """
    matrix = np.random.rand(rows, cols).astype(np.float32)
    return matrix.tolist()  # Convert numpy array to a Python nested list for JSON serialization

def create_json_file(file_name: str, d_model: int, sequence_len: int):
    """
    Creates a JSON file containing matrices for queries, keys, and values.

    Parameters:
    - file_name (str): The name of the JSON file to be created.
    - d_model (int): Dimensionality of each vector (number of features per vector).
    - sequence_len (int): Number of vectors in the sequence.
    """
    # Generate matrices for queries, keys, and values
    data = {
        "queries": generate_matrix(sequence_len, d_model),
        "keys": generate_matrix(sequence_len, d_model),
        "values": generate_matrix(sequence_len, d_model)
    }
    
    # Write the data to a JSON file
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"JSON file '{file_name}' created with matrices for queries, keys, and values.")

def main():
    if len(sys.argv) != 4:
        print("Usage: python create_json_matrices.py <file_name.json> <d_model> <sequence_len>")
        sys.exit(1)
    
    # Get the input arguments
    file_name = sys.argv[1]
    d_model = int(sys.argv[2])
    sequence_len = int(sys.argv[3])
    
    # Create the JSON file
    create_json_file(file_name, d_model, sequence_len)

if __name__ == "__main__":
    main()
