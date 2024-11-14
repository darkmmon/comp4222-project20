import json
import numpy as np
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
import os
import math

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

def read_json_file(path = 'data_preprocess/Subscription_Boxes.jsonl'):
    """
    Reads the JSONL file and returns a NumPy array of review records.

    Each review record contains:
    [rating, length_of_title, length_of_text, asin, user_id, timestamp, helpful_vote, verified_purchase]

    Args:
    - path (str): Path to the JSONL file.

    Returns:
    - np.ndarray: Array of review records.
    """
    matrix = []

    with open(path, 'r') as file:
        for line in tqdm(file, desc="Reading JSON Lines"):
            line = line.strip()
            if line:
                try:
                    json_obj = json.loads(line)
                    
                    rating = json_obj.get('rating')
                    title = json_obj.get('title', '')
                    text = json_obj.get('text', '')
                    length_of_title = len(title)
                    length_of_text = len(text)
                    asin = json_obj.get("asin")
                    user_id = json_obj.get('user_id')
                    timestamp = json_obj.get('timestamp')
                    helpful_vote = json_obj.get('helpful_vote', 0)
                    verified_purchase = json_obj.get('verified_purchase', False)

                    # Ensure all necessary fields are present
                    if None in [rating, asin, user_id, timestamp]:
                        continue  # Skip incomplete records

                    matrix.append((rating, length_of_title, length_of_text, asin, user_id, timestamp, helpful_vote, verified_purchase))
                except json.JSONDecodeError:
                    continue  # Skip malformed lines

    return np.array(matrix, dtype=object)       # [rating, length_of_title, length_of_text, asin, user_id, timestamp, helpful_vote, verified_purchase] * 15237

def assign_products_to_buyers(matrix):
    """
    Assigns products to buyers.

    Args:
    - matrix (np.ndarray): Array of review records.

    Returns:
    - dict: Mapping from buyer_id to a set of product_ids they've reviewed.
    """
    buyer_with_products = {}

    for row in matrix:
        rating = row[0]
        length_of_title = row[1]
        length_of_text = row[2]
        asin = row[3]
        user_id = row[4]
        timestamp = row[5]
        helpful_vote = row[6]
        verified_purchase = row[7]

        # Initialize set if buyer not present
        if user_id not in buyer_with_products:
            buyer_with_products[user_id] = set()

        # Add the product to the buyer's set
        buyer_with_products[user_id].add(asin)

    return buyer_with_products

def create_UPU_matrix(buyer_n_products):
    """
    Creates the UPU adjacency matrix where users are connected if they've reviewed at least one same product.

    Args:
    - buyer_n_products (dict): Mapping from buyer_id to a set of product_ids.

    Returns:
    - np.ndarray: UPU adjacency matrix.
    - list: List of unique buyers corresponding to matrix indices.
    """
    unique_buyers = list(buyer_n_products.keys())    # Convert dictionary keys to list
    num_buyers = len(unique_buyers)

    # Create a mapping from buyer_id to index
    buyer_to_idx = {buyer: idx for idx, buyer in enumerate(unique_buyers)}

    # Invert the mapping to get product_to_buyers
    product_to_buyers = defaultdict(set)
    for buyer, products in buyer_n_products.items():
        for product in products:
            product_to_buyers[product].add(buyer)

    # Initialize adjacency matrix
    adjacency_matrix = np.zeros((num_buyers, num_buyers), dtype=int)

    # Populate adjacency matrix based on shared products
    for buyers in tqdm(product_to_buyers.values(), desc="Processing UPU Connections"):
        if len(buyers) < 2:
            continue  # No edges to add
        # Convert buyer IDs to indices
        buyer_indices = [buyer_to_idx[buyer] for buyer in buyers]
        # Generate all unique pairs and set adjacency
        for i, j in combinations(buyer_indices, 2):
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1  # Ensure symmetry

    return adjacency_matrix, unique_buyers

def create_USV_matrix(matrix, buyer_to_idx):
    """
    Creates the USV adjacency matrix where users are connected if they've given the same rating
    to the same product within one week.

    Args:
    - matrix (np.ndarray): Array of review records.
    - buyer_to_idx (dict): Mapping from buyer_id to index.

    Returns:
    - np.ndarray: USV adjacency matrix.
    """
    # Organize reviews per product and rating
    product_rating_to_reviews = defaultdict(list)
    for row in matrix:
        rating = row[0]
        asin = row[3]
        user_id = row[4]
        timestamp = row[5]
        helpful_vote = row[6]
        verified_purchase = row[7]

        # Only consider verified purchases for USV (optional)
        if not verified_purchase:
            continue

        product_rating_to_reviews[(asin, rating)].append((user_id, timestamp))

    # Initialize USV adjacency matrix
    num_buyers = len(buyer_to_idx)
    usv_matrix = np.zeros((num_buyers, num_buyers), dtype=int)

    # Define one week in milliseconds
    one_week_millis = 7 * 24 * 60 * 60 * 1000

    # Populate USV adjacency matrix
    for (asin, rating), reviews in tqdm(product_rating_to_reviews.items(), desc="Processing USV Connections"):
        if len(reviews) < 2:
            continue  # No edges to add
        # Sort reviews by timestamp
        reviews_sorted = sorted(reviews, key=lambda x: x[1])
        n = len(reviews_sorted)
        for i in range(n):
            user_i, timestamp_i = reviews_sorted[i]
            for j in range(i + 1, n):
                user_j, timestamp_j = reviews_sorted[j]
                # Check if within one week
                if timestamp_j - timestamp_i > one_week_millis:
                    break  # No need to check further as list is sorted
                # Get indices
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
    homo_matrix = np.maximum(upu_matrix, usv_matrix)  # Element-wise maximum
    return homo_matrix

def create_feature_matrix(matrix, buyer_to_idx, unique_buyers):
    """
    Creates the feature matrix for each buyer.

    Features:
    1. Number of rated products
    2. Average length of title name
    3. Average length of text
    4. Entropy of rating
    5. Total number of helpful votes

    Args:
    - matrix (np.ndarray): Array of review records.
    - buyer_to_idx (dict): Mapping from buyer_id to index.
    - unique_buyers (list): List of unique buyer_ids.

    Returns:
    - np.ndarray: Feature matrix of shape [num_buyers, 5].
    """
    num_buyers = len(unique_buyers)
    feature_matrix = np.zeros((num_buyers, 5), dtype=float)

    # Initialize dictionaries to hold feature data
    buyer_rated_products = defaultdict(set)
    buyer_title_lengths = defaultdict(list)
    buyer_text_lengths = defaultdict(list)
    buyer_ratings = defaultdict(list)
    buyer_helpful_votes = defaultdict(int)

    # Populate the dictionaries
    for row in matrix:
        rating = row[0]
        length_of_title = row[1]
        length_of_text = row[2]
        asin = row[3]
        user_id = row[4]
        timestamp = row[5]
        helpful_vote = row[6]
        verified_purchase = row[7]

        if user_id not in buyer_to_idx:
            continue  # Skip users not in the buyer_to_idx mapping

        idx = buyer_to_idx[user_id]
        buyer_rated_products[idx].add(asin)
        buyer_title_lengths[idx].append(length_of_title)
        buyer_text_lengths[idx].append(length_of_text)
        buyer_ratings[idx].append(rating)
        buyer_helpful_votes[idx] += helpful_vote

    # Compute features
    for idx in range(num_buyers):
        # Feature 1: Number of rated products
        feature_matrix[idx, 0] = len(buyer_rated_products[idx])

        # Feature 2: Average length of title name
        if buyer_title_lengths[idx]:
            feature_matrix[idx, 1] = np.mean(buyer_title_lengths[idx])
        else:
            feature_matrix[idx, 1] = 0.0

        # Feature 3: Average length of text
        if buyer_text_lengths[idx]:
            feature_matrix[idx, 2] = np.mean(buyer_text_lengths[idx])
        else:
            feature_matrix[idx, 2] = 0.0

        # Feature 4: Entropy of rating
        ratings = buyer_ratings[idx]
        if ratings:
            entropy = compute_entropy(ratings)
            feature_matrix[idx, 3] = entropy
        else:
            feature_matrix[idx, 3] = 0.0

        # Feature 5: Total number of helpful votes
        feature_matrix[idx, 4] = buyer_helpful_votes[idx]

    return feature_matrix

def compute_entropy(ratings):
    """
    Computes the entropy of the rating distribution for a buyer.

    Args:
    - ratings (list): List of ratings given by the buyer.

    Returns:
    - float: Entropy value.
    """
    rating_counts = defaultdict(int)
    for rating in ratings:
        rating_counts[rating] += 1

    total = len(ratings)
    entropy = 0.0
    for count in rating_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p, 2)
    return entropy

def create_label_matrix(feature_matrix):
    """
    Creates the label matrix based on total helpful votes.

    Label:
    - 1: If total helpful votes > 75% of average helpful votes across all buyers.
    - 0: Otherwise.

    Args:
    - feature_matrix (np.ndarray): Feature matrix of shape [num_buyers, 5].

    Returns:
    - np.ndarray: Label matrix of shape [num_buyers, 1].
    """
    total_helpful_votes = feature_matrix[:, 4]
    average_helpful = np.mean(total_helpful_votes)
    threshold = 0.75 * average_helpful

    labels = (total_helpful_votes > threshold).astype(int).reshape(-1, 1)
    return labels

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

def save_feature_matrix(feature_matrix, path):
    """
    Saves the feature matrix to a .txt file.

    Each row of the feature matrix is saved as a line with elements separated by spaces.

    Args:
    - feature_matrix (np.ndarray): Feature matrix of shape [num_buyers, 5].
    - path (str): File path to save the feature matrix.
    """
    np.savetxt(path, feature_matrix, fmt='%.4f', delimiter=' ')
    print(f"Saved feature matrix to {path}")

def save_label_matrix(label_matrix, path):
    """
    Saves the label matrix to a .txt file.

    Each label is saved as a separate line.

    Args:
    - label_matrix (np.ndarray): Label matrix of shape [num_buyers, 1].
    - path (str): File path to save the label matrix.
    """
    np.savetxt(path, label_matrix, fmt='%d', delimiter=' ')
    print(f"Saved label matrix to {path}")

def main():
    # Paths
    json_path = 'data_preprocess_matrix/Subscription_Boxes.jsonl'
    adjacency_save_dir = 'adjacency_matrices'
    feature_save_path = 'feature_matrix.txt'
    label_save_path = 'label_matrix.txt'
    buyer_to_idx_path = 'buyer_to_idx.json'

    # Create save directory if it doesn't exist
    os.makedirs(adjacency_save_dir, exist_ok=True)

    # Step 1: Read JSONL data
    matrix = read_json_file(json_path)

    # Step 2: Assign products to buyers (for UPU)
    buyers_assigned_products = assign_products_to_buyers(matrix)   #(total 15327 buyers (nodes))

    # Step 3: Create UPU adjacency matrix
    upu_matrix, unique_buyers = create_UPU_matrix(buyers_assigned_products)
    print("UPU Matrix Shape:", upu_matrix.shape)
    print("Number of Unique Buyers:", len(unique_buyers))     #1128 node(buyers)

    # Step 4: Create buyer-to-index mapping and save it
    buyer_to_idx = {buyer: idx for idx, buyer in enumerate(unique_buyers)}
    with open(buyer_to_idx_path, 'w') as f:
        json.dump(buyer_to_idx, f)
    print(f"Saved buyer_to_idx mapping to {buyer_to_idx_path}")

    # Step 5: Create USV adjacency matrix
    usv_matrix = create_USV_matrix(matrix, buyer_to_idx)
    print("USV Matrix Shape:", usv_matrix.shape)

    # Step 6: Create HOMO adjacency matrix
    homo_matrix = create_HOMO_matrix(upu_matrix, usv_matrix)
    print("HOMO Matrix Shape:", homo_matrix.shape)

    # Step 7: Save Adjacency Matrices as .txt Files
    save_matrix_to_txt(upu_matrix, os.path.join(adjacency_save_dir, 'UPU_adj_matrix.txt'))
    save_matrix_to_txt(usv_matrix, os.path.join(adjacency_save_dir, 'USV_adj_matrix.txt'))
    save_matrix_to_txt(homo_matrix, os.path.join(adjacency_save_dir, 'Homo_adj_matrix.txt'))

    # Step 8: Create Feature Matrix
    feature_matrix = create_feature_matrix(matrix, buyer_to_idx, unique_buyers)
    print("Feature Matrix Shape:", feature_matrix.shape)

    # Step 9: Create Label Matrix
    label_matrix = create_label_matrix(feature_matrix)
    print("Label Matrix Shape:", label_matrix.shape)

    # Step 10: Save Feature and Label Matrices
    save_feature_matrix(feature_matrix, feature_save_path)
    save_label_matrix(label_matrix, label_save_path)

    # Step 11: (Optional) Verify Matrices
    # You can implement verification here if needed

if __name__ == '__main__':
    main()