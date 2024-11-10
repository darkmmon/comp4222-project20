import numpy as np

# input : dictionay contain buyers and its bought product

def create_UPU_matrix(buyer_to_products):
    # Get the list of unique buyers
    unique_buyers = list(buyer_to_products.keys())
    num_buyers = len(unique_buyers)

    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((num_buyers, num_buyers), dtype=int)

    # Create a mapping from buyer IDs to matrix indices
    buyer_to_idx = {buyer: idx for idx, buyer in enumerate(unique_buyers)}

    # Loop through all pairs of buyers and check if they share any products
    for i in range(num_buyers):
        for j in range(i + 1, num_buyers):  # Avoid self-connections and redundant checks
            buyer_i = unique_buyers[i]
            buyer_j = unique_buyers[j]

            # Check if buyer_i and buyer_j share any products
            if buyer_to_products[buyer_i] & buyer_to_products[buyer_j]:  # Intersection of product sets
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1  # Ensure symmetry

    return adjacency_matrix, unique_buyers
