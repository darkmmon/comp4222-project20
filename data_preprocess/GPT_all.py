import json
from collections import defaultdict
from itertools import combinations
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import pandas as pd
import networkx as nx
import os
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
ONE_WEEK_IN_MILLIS = 7 * 24 * 60 * 60 * 1000  # 7 days in milliseconds

def read_json_file(path='data_preprocess/Subscription_Boxes.jsonl'):
    """
    Reads the JSONL file and returns a list of review records.

    Args:
    - path (str): Path to the JSONL file.

    Returns:
    - list of tuples: Each tuple contains relevant fields extracted from the JSON object.
    """
    logging.info(f"Reading data from {path}...")
    matrix = []
    with open(path, 'r') as file:
        for line in tqdm(file, desc="Reading JSON Lines"):
            line = line.strip()
            if line:
                try:
                    json_obj = json.loads(line)
                    rating = json_obj.get('rating')
                    length_of_title = len(json_obj.get('title', ''))
                    length_of_text = len(json_obj.get('text', ''))
                    asin = json_obj.get("asin")
                    user_id = json_obj.get('user_id')
                    timestamp = json_obj.get('timestamp')
                    helpful_vote = json_obj.get('helpful_vote')
                    verified_purchase = json_obj.get('verified_purchase')

                    # Ensure all necessary fields are present
                    if None in [rating, asin, user_id, timestamp]:
                        continue  # Skip incomplete records

                    matrix.append((rating, length_of_title, length_of_text, asin, user_id, timestamp, helpful_vote, verified_purchase))
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
    logging.info(f"Total reviews parsed: {len(matrix)}")
    return matrix  # List of tuples

def create_buyer_to_idx(matrix):
    """
    Creates a mapping from buyer IDs to unique indices.

    Args:
    - matrix (list of tuples): Parsed review data.

    Returns:
    - dict: Mapping from user_id to index.
    - list: Sorted list of unique user_ids.
    """
    logging.info("Creating buyer to index mapping...")
    unique_buyers = sorted({row[4] for row in matrix})  # Sort for consistency
    buyer_to_idx = {buyer: idx for idx, buyer in enumerate(unique_buyers)}
    logging.info(f"Total unique buyers: {len(unique_buyers)}")
    return buyer_to_idx, unique_buyers

def build_UPU_adjacency(matrix, buyer_to_idx):
    """
    Builds the UPU adjacency matrix where users are connected if they've reviewed at least one same product.

    Args:
    - matrix (list of tuples): Parsed review data.
    - buyer_to_idx (dict): Mapping from user_id to index.

    Returns:
    - scipy.sparse.csr_matrix: Sparse adjacency matrix for UPU.
    """
    logging.info("Building UPU adjacency matrix...")
    product_to_buyers = defaultdict(set)
    for row in matrix:
        asin = row[3]
        user_id = row[4]
        product_to_buyers[asin].add(user_id)

    num_buyers = len(buyer_to_idx)
    adj_matrix = sp.lil_matrix((num_buyers, num_buyers), dtype=int)

    for buyers in tqdm(product_to_buyers.values(), desc="Processing UPU Connections"):
        if len(buyers) < 2:
            continue  # No edges to add
        # Convert user_ids to indices
        buyer_indices = [buyer_to_idx[buyer] for buyer in buyers if buyer in buyer_to_idx]
        # Create all possible unique pairs and set adjacency
        for i, j in combinations(buyer_indices, 2):
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1  # Ensure symmetry

    logging.info("Converting UPU adjacency matrix to CSR format...")
    return adj_matrix.tocsr()

def build_USV_adjacency(matrix, buyer_to_idx):
    """
    Builds the USV adjacency matrix where users are connected if they've given the same rating to the same product within one week.

    Args:
    - matrix (list of tuples): Parsed review data.
    - buyer_to_idx (dict): Mapping from user_id to index.

    Returns:
    - scipy.sparse.csr_matrix: Sparse adjacency matrix for USV.
    """
    logging.info("Building USV adjacency matrix...")
    product_rating_time_to_buyers = defaultdict(list)
    
    # Group users by (asin, rating)
    for row in matrix:
        asin = row[3]
        user_id = row[4]
        rating = row[0]
        timestamp = row[5]
        product_rating_time_to_buyers[(asin, rating)].append((user_id, timestamp))

    num_buyers = len(buyer_to_idx)
    adj_matrix = sp.lil_matrix((num_buyers, num_buyers), dtype=int)
    
    for (asin, rating), users in tqdm(product_rating_time_to_buyers.items(), desc="Processing USV Groups"):
        if len(users) < 2:
            continue  # No edges to add
        # Sort users by timestamp to optimize the time window checks
        users_sorted = sorted(users, key=lambda x: x[1])
        n = len(users_sorted)
        for i in range(n):
            user_i, timestamp_i = users_sorted[i]
            for j in range(i + 1, n):
                user_j, timestamp_j = users_sorted[j]
                if timestamp_j - timestamp_i > ONE_WEEK_IN_MILLIS:
                    break  # Beyond the one-week window
                # Add edge if both users are in the mapping
                if user_i in buyer_to_idx and user_j in buyer_to_idx:
                    idx_i = buyer_to_idx[user_i]
                    idx_j = buyer_to_idx[user_j]
                    adj_matrix[idx_i, idx_j] = 1
                    adj_matrix[idx_j, idx_i] = 1  # Ensure symmetry

    logging.info("Converting USV adjacency matrix to CSR format...")
    return adj_matrix.tocsr()

def build_HOMO_adjacency(upu_matrix, usv_matrix):
    """
    Builds the HOMO adjacency matrix as the union of UPU and USV adjacency matrices.

    Args:
    - upu_matrix (scipy.sparse.csr_matrix): UPU adjacency matrix.
    - usv_matrix (scipy.sparse.csr_matrix): USV adjacency matrix.

    Returns:
    - scipy.sparse.csr_matrix: Sparse adjacency matrix for HOMO.
    """
    logging.info("Building HOMO adjacency matrix as the union of UPU and USV...")
    Homo_matrix = upu_matrix.maximum(usv_matrix)  # Element-wise maximum ensures union
    logging.info("HOMO adjacency matrix created.")
    return Homo_matrix

def save_adjacency_matrix_txt(adj_matrix, path):
    """
    Saves the adjacency matrix as a .txt edge list file.

    Each line in the file represents an undirected edge in the format:
    source_index target_index

    Args:
    - adj_matrix (scipy.sparse.csr_matrix): Adjacency matrix to save.
    - path (str): File path to save the adjacency matrix.
    """
    logging.info(f"Saving adjacency matrix to {path}...")
    adj_matrix_coo = adj_matrix.tocoo()
    with open(path, 'w') as f:
        for i, j in zip(adj_matrix_coo.row, adj_matrix_coo.col):
            if i < j:  # To ensure each edge is written only once
                f.write(f"{i} {j}\n")
    logging.info("Adjacency matrix saved.")

def save_mapping(buyer_to_idx, path):
    """
    Saves the buyer_to_idx mapping as a JSON file.

    Args:
    - buyer_to_idx (dict): Mapping from user_id to index.
    - path (str): File path to save the mapping.
    """
    logging.info(f"Saving buyer_to_idx mapping to {path}...")
    with open(path, 'w') as f:
        json.dump(buyer_to_idx, f)
    logging.info("Buyer to index mapping saved.")

def verify_matrices_alignment(upu_matrix, usv_matrix, homo_matrix, unique_buyers):
    """
    Verifies that the adjacency matrices are correctly aligned and populated.

    Args:
    - upu_matrix (scipy.sparse.csr_matrix): UPU adjacency matrix.
    - usv_matrix (scipy.sparse.csr_matrix): USV adjacency matrix.
    - homo_matrix (scipy.sparse.csr_matrix): HOMO adjacency matrix.
    - unique_buyers (list): List of unique user_ids.

    Returns:
    - None
    """
    logging.info("Verifying adjacency matrices alignment and integrity...")

    # Check shapes
    assert upu_matrix.shape == usv_matrix.shape == homo_matrix.shape, "Adjacency matrices have different shapes."

    # Check symmetry
    assert (upu_matrix != upu_matrix.transpose()).nnz == 0, "UPU matrix is not symmetric."
    assert (usv_matrix != usv_matrix.transpose()).nnz == 0, "USV matrix is not symmetric."
    assert (homo_matrix != homo_matrix.transpose()).nnz == 0, "HOMO matrix is not symmetric."

    # Check that HOMO is the union
    difference = homo_matrix - upu_matrix.maximum(usv_matrix)
    assert difference.nnz == 0, "HOMO matrix is not the union of UPU and USV matrices."

    # Spot-checking a few users
    sample_indices = np.random.choice(len(unique_buyers), size=5, replace=False)
    for idx in sample_indices:
        buyer = unique_buyers[idx]
        upu_connections = upu_matrix[idx].nonzero()[1]
        usv_connections = usv_matrix[idx].nonzero()[1]
        homo_connections = homo_matrix[idx].nonzero()[1]
        # Check if Homo connections are the union
        expected = np.union1d(upu_connections, usv_connections)
        if not np.array_equal(np.sort(homo_connections), expected):
            logging.warning(f"Mismatch in HOMO connections for buyer {buyer}.")
        else:
            logging.info(f"Buyer {buyer}: UPU = {len(upu_connections)}, USV = {len(usv_connections)}, HOMO = {len(homo_connections)}")

    logging.info("Verification completed successfully.")

def main():
    # File paths
    JSON_PATH = 'data_preprocess/Subscription_Boxes.jsonl'
    UPU_SAVE_PATH = 'adjacency_matrices/UPU_adj_matrix.txt'
    USV_SAVE_PATH = 'adjacency_matrices/USV_adj_matrix.txt'
    HOMO_SAVE_PATH = 'adjacency_matrices/Homo_adj_matrix.txt'
    MAPPING_SAVE_PATH = 'mappings/buyer_to_idx.json'

    # Create directories if they don't exist
    os.makedirs('adjacency_matrices', exist_ok=True)
    os.makedirs('mappings', exist_ok=True)

    # Step 1: Read and parse JSON data
    matrix = read_json_file(JSON_PATH)

    # Step 2: Create a unified buyer to index mapping
    buyer_to_idx, unique_buyers = create_buyer_to_idx(matrix)

    # Step 3: Build UPU adjacency matrix
    upu_matrix = build_UPU_adjacency(matrix, buyer_to_idx)
    logging.info(f"UPU Adjacency Matrix Shape: {upu_matrix.shape}")

    # Step 4: Build USV adjacency matrix
    usv_matrix = build_USV_adjacency(matrix, buyer_to_idx)
    logging.info(f"USV Adjacency Matrix Shape: {usv_matrix.shape}")

    # Step 5: Build HOMO adjacency matrix
    homo_matrix = build_HOMO_adjacency(upu_matrix, usv_matrix)
    logging.info(f"HOMO Adjacency Matrix Shape: {homo_matrix.shape}")

    # Step 6: Save adjacency matrices as .txt files
    save_adjacency_matrix_txt(upu_matrix, UPU_SAVE_PATH)
    save_adjacency_matrix_txt(usv_matrix, USV_SAVE_PATH)
    save_adjacency_matrix_txt(homo_matrix, HOMO_SAVE_PATH)

    # Step 7: Save buyer_to_idx mapping
    save_mapping(buyer_to_idx, MAPPING_SAVE_PATH)

    # Step 8: Verification
    verify_matrices_alignment(upu_matrix, usv_matrix, homo_matrix, unique_buyers)

    logging.info("All adjacency matrices and mappings have been successfully created and saved.")

if __name__ == "__main__":
    main()