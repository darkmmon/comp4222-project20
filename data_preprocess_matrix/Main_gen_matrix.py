import json
import numpy as np
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
import os

def print_dict(buyer_dict, index):
    # Convert the keys to a list
    keys_list = list(buyer_dict.keys())

    # Check if there are at least (index + 1) keys
    if len(keys_list) > index:  # Check for the requested index
        key_ = keys_list[index]  # Get the key at the specified index
        elements_ = buyer_dict[key_]  # Get the associated products
        print(f"{key_}: {elements_}")  # Print the key and its associated products
    else:
        print(f"The dictionary does not have {index + 1} keys.")  # Adjusted for 1-based indexing

def read_json_file(path='data_preprocess_matrix/Subscription_Boxes.jsonl'):
    matrix = []

    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                json_obj = json.loads(line)
                
                rating = json_obj.get('rating')
                asin = json_obj.get("asin")
                user_id = json_obj.get('user_id')
                timestamp = json_obj.get('timestamp')                

                matrix.append((rating, asin, user_id, timestamp))
                
    return np.array(matrix, dtype=object)       #[rating, product_ID, user_ID, timestamp] * 16216 (buy record, not user)

def assign_products_to_buyers(matrix):    # for UPU
    buyer_with_products = {}

    for row in matrix:
        product_id = row[1]      # Product ID (asin)
        buyer_id = row[2]        # Buyer ID

        # If the buyer is not in the dictionary, initialize an empty set for their products
        if buyer_id not in buyer_with_products:
            buyer_with_products[buyer_id] = set()

        # Add the product to the buyer's set of products
        buyer_with_products[buyer_id].add(product_id)
    
    # print(len(buyer_with_products))
    # print_dict(buyer_with_products, 9999)
    return buyer_with_products

def create_UPU_matrix(buyer_n_products):
    """
    Creates the UPU adjacency matrix where users are connected if they've reviewed at least one same product.

    Args:
    - buyer_n_products (dict): A dictionary mapping buyer_id to a set of product_ids.

    Returns:
    - np.ndarray: The UPU adjacency matrix (square matrix with shape [num_buyers, num_buyers]).
    - list: The list of unique buyers in the same order as the adjacency matrix.
    """
    # Get the list of unique buyers
    unique_buyers = list(buyer_n_products.keys())    # Convert keys in dictionary to list
    num_buyers = len(unique_buyers)

    # Create a buyer to index mapping to maintain consistent ordering
    buyer_to_idx = {buyer: idx for idx, buyer in enumerate(unique_buyers)}

    # Invert the buyer_n_products to get product_to_buyers mapping
    product_to_buyers = defaultdict(set)
    for buyer, products in buyer_n_products.items():
        for product in products:
            product_to_buyers[product].add(buyer)

    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((num_buyers, num_buyers), dtype=int)

    # Iterate over each product and connect all pairs of buyers who have reviewed it
    for buyers in tqdm(product_to_buyers.values(), desc="Processing UPU Connections"):
        if len(buyers) < 2:
            continue  # No edges to add for single buyer
        # Convert buyer IDs to indices
        buyer_indices = [buyer_to_idx[buyer] for buyer in buyers]
        # Generate all unique pairs of buyers
        for i, j in combinations(buyer_indices, 2):
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1  # Ensure symmetry

    return adjacency_matrix, unique_buyers  # Return the adjacency matrix and the list of unique buyers

def create_buyer_to_idx(unique_buyers):
    """
    Creates a mapping from buyer IDs to unique indices.

    Args:
    - unique_buyers (list): List of unique user IDs.

    Returns:
    - dict: Mapping from user_id to index.
    """
    buyer_to_idx = {buyer_id: idx for idx, buyer_id in enumerate(unique_buyers)}
    return buyer_to_idx

def assign_ratings_to_users(matrix):
    """
    Assigns ratings and timestamps to users per product.

    Args:
    - matrix (np.ndarray): Parsed review data.

    Returns:
    - dict: Mapping from product_id to a list of tuples (user_id, rating, timestamp).
    """
    product_with_users = defaultdict(list)

    for row in matrix:
        rating = row[0]        # Rating
        asin = row[1]          # Product ID (asin)
        user_id = row[2]       # User ID
        timestamp = row[3]     # Timestamp

        # Append (user_id, rating, timestamp) to the product's list
        product_with_users[asin].append((user_id, rating, timestamp))
    
    return product_with_users

def create_USV_matrix(product_with_users, buyer_to_idx):
    """
    Creates the USV adjacency matrix where users are connected if they've given the same rating
    to the same product within one week.

    Args:
    - product_with_users (dict): Mapping from product_id to list of tuples (user_id, rating, timestamp).
    - buyer_to_idx (dict): Mapping from user_id to unique index.

    Returns:
    - np.ndarray: USV adjacency matrix.
    """
    num_buyers = len(buyer_to_idx)
    usv_matrix = np.zeros((num_buyers, num_buyers), dtype=int)

    for asin, users in tqdm(product_with_users.items(), desc="Processing USV Connections"):
        # Group users by rating
        rating_groups = defaultdict(list)
        for user_id, rating, timestamp in users:
            rating_groups[rating].append((user_id, timestamp))
        
        for rating, user_list in rating_groups.items():
            # Sort the user_list by timestamp to optimize the within-one-week check
            user_list_sorted = sorted(user_list, key=lambda x: x[1])
            n = len(user_list_sorted)
            for i in range(n):
                user_i, timestamp_i = user_list_sorted[i]
                for j in range(i + 1, n):
                    user_j, timestamp_j = user_list_sorted[j]
                    if timestamp_j - timestamp_i > 7 * 24 * 60 * 60 * 1000:  # One week in milliseconds
                        break  # No need to check further as the list is sorted
                    # Connect the users
                    idx_i = buyer_to_idx.get(user_i)
                    idx_j = buyer_to_idx.get(user_j)
                    if idx_i is not None and idx_j is not None:
                        usv_matrix[idx_i, idx_j] = 1
                        usv_matrix[idx_j, idx_i] = 1  # Ensure symmetry

    return usv_matrix

def create_HOMO_matrix(upu_matrix, usv_matrix):
    """
    Creates the HOMO adjacency matrix as the union of UPU and USV matrices.

    Args:
    - upu_matrix (np.ndarray): UPU adjacency matrix.
    - usv_matrix (np.ndarray): USV adjacency matrix.

    Returns:
    - np.ndarray: HOMO adjacency matrix.
    """
    homo_matrix = np.maximum(upu_matrix, usv_matrix)  # Element-wise maximum to combine connections
    return homo_matrix

def save_matrix_to_txt(matrix, path):
    """
    Saves a square adjacency matrix to a .txt file.

    Each row of the matrix is saved as a line with elements separated by spaces.

    Args:
    - matrix (np.ndarray): Square adjacency matrix.
    - path (str): File path to save the matrix.
    """
    np.savetxt(path, matrix, fmt='%d', delimiter=' ')
    print(f"Saved matrix to {path}")

def verify_matrices_alignment(upu_matrix, usv_matrix, homo_matrix, unique_buyers):
    """
    Verifies that the adjacency matrices are correctly aligned and populated.

    Args:
    - upu_matrix (np.ndarray): UPU adjacency matrix.
    - usv_matrix (np.ndarray): USV adjacency matrix.
    - homo_matrix (np.ndarray): HOMO adjacency matrix.
    - unique_buyers (list): List of unique user_ids.

    Returns:
    - None
    """
    print("Verifying adjacency matrices alignment and integrity...")

    # Check shapes
    if upu_matrix.shape != usv_matrix.shape or usv_matrix.shape != homo_matrix.shape:
        print("Error: Adjacency matrices have different shapes.")
        return
    
    # Check symmetry
    if not np.array_equal(upu_matrix, upu_matrix.T):
        print("Error: UPU matrix is not symmetric.")
    if not np.array_equal(usv_matrix, usv_matrix.T):
        print("Error: USV matrix is not symmetric.")
    if not np.array_equal(homo_matrix, homo_matrix.T):
        print("Error: HOMO matrix is not symmetric.")
    
    # Check that HOMO is the union
    if not np.array_equal(homo_matrix, np.maximum(upu_matrix, usv_matrix)):
        print("Error: HOMO matrix is not the union of UPU and USV matrices.")
    else:
        print("HOMO matrix is correctly the union of UPU and USV matrices.")
    
    # Spot-checking a few users
    sample_indices = np.random.choice(len(unique_buyers), size=5, replace=False)
    for idx in sample_indices:
        buyer = unique_buyers[idx]
        upu_connections = upu_matrix[idx]
        usv_connections = usv_matrix[idx]
        homo_connections = homo_matrix[idx]
        # The homogenous connections should be the logical OR of upu and usv
        expected = np.maximum(upu_connections, usv_connections)
        actual = homo_connections
        if not np.array_equal(actual, expected):
            print(f"Mismatch in HOMO connections for buyer {buyer}.")
        else:
            num_upu = np.sum(upu_connections)
            num_usv = np.sum(usv_connections)
            num_homo = np.sum(homo_connections)
            print(f"Buyer {buyer}: UPU Connections = {num_upu}, USV Connections = {num_usv}, HOMO Connections = {num_homo}")

    print("Verification completed.")

if __name__ == '__main__':
    matrix = read_json_file()

    # Step 1: Assign products to buyers (UPU)
    buyers_assigned_products = assign_products_to_buyers(matrix)   #(total 15327 buyers (nodes))
    
    # Step 2: Create UPU adjacency matrix
    upu_matrix, unique_buyers = create_UPU_matrix(buyers_assigned_products)
    
    # Step 3: Create buyer-to-index mapping
    buyer_to_idx = {buyer: idx for idx, buyer in enumerate(unique_buyers)}
    
    print("UPU Matrix Shape:", upu_matrix.shape)
    print("Number of Unique Buyers:", len(unique_buyers))
    
    # Step 4: Assign ratings to users (for USV)
    product_with_users = assign_ratings_to_users(matrix)
    
    # Step 5: Create USV adjacency matrix
    usv_matrix = create_USV_matrix(product_with_users, buyer_to_idx)
    print("USV Matrix Shape:", usv_matrix.shape)
    
    # Step 6: Create HOMO adjacency matrix
    homo_matrix = create_HOMO_matrix(upu_matrix, usv_matrix)
    print("HOMO Matrix Shape:", homo_matrix.shape)
    
    # Step 7: Save all matrices as square .txt files
    save_dir = 'adjacency_matrices'
    os.makedirs(save_dir, exist_ok=True)
    save_matrix_to_txt(upu_matrix, os.path.join(save_dir, 'UPU_adj_matrix.txt'))
    save_matrix_to_txt(usv_matrix, os.path.join(save_dir, 'USV_adj_matrix.txt'))
    save_matrix_to_txt(homo_matrix, os.path.join(save_dir, 'Homo_adj_matrix.txt'))
    
    # Step 8: (Optional) Save the buyer_to_idx mapping
    mapping_path = 'buyer_to_idx.json'
    with open(mapping_path, 'w') as f:
        json.dump(buyer_to_idx, f)
    print(f"Saved buyer_to_idx mapping to {mapping_path}")
    
    # Step 9: Verify the matrices
    verify_matrices_alignment(upu_matrix, usv_matrix, homo_matrix, unique_buyers)