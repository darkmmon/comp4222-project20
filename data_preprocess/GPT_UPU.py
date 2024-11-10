import json
from collections import defaultdict
import networkx as nx
from itertools import combinations
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm

def read_json_file(path='data_preprocess/Subscription_Boxes.jsonl'):
    buyer_with_products = defaultdict(set)

    with open(path, 'r') as file:
        for line in tqdm(file, desc="Reading JSON lines"):
            line = line.strip()
            if not line:
                continue
            try:
                json_obj = json.loads(line)
                asin = json_obj.get("asin")
                user_id = json_obj.get('user_id')
                if asin and user_id:
                    buyer_with_products[user_id].add(asin)
            except json.JSONDecodeError:
                continue  # Handle or log malformed lines as needed

    return buyer_with_products

def create_UPU_graph(buyer_n_products):
    G = nx.Graph()
    G.add_nodes_from(buyer_n_products.keys())

    # Inverted index: product -> set of buyers
    product_to_buyers = defaultdict(set)
    for buyer, products in tqdm(buyer_n_products.items(), desc="Building product to buyers mapping"):
        for product in products:
            product_to_buyers[product].add(buyer)

    # Add edges based on shared products
    for buyers in tqdm(product_to_buyers.values(), desc="Adding edges for shared products"):
        if len(buyers) < 2:
            continue
        for buyer1, buyer2 in combinations(buyers, 2):
            G.add_edge(buyer1, buyer2)

    return G

def graph_to_sparse_adjacency_matrix(G, unique_buyers):
    buyer_to_idx = {buyer: idx for idx, buyer in enumerate(unique_buyers)}
    adj_matrix = sp.lil_matrix((len(unique_buyers), len(unique_buyers)), dtype=int)

    for buyer1, buyer2 in tqdm(G.edges(), desc="Mapping edges to matrix indices"):
        idx1 = buyer_to_idx[buyer1]
        idx2 = buyer_to_idx[buyer2]
        adj_matrix[idx1, idx2] = 1
        adj_matrix[idx2, idx1] = 1  # Ensure symmetry

    return adj_matrix.tocsr()

def main():
    # Step 1: Read and parse JSON data
    buyer_to_products = read_json_file()

    # Step 2: Create UPU graph
    G = create_UPU_graph(buyer_to_products)

    # Step 3: Convert graph to adjacency matrix
    unique_buyers = list(buyer_to_products.keys())
    adj_matrix = graph_to_sparse_adjacency_matrix(G, unique_buyers)

    # Step 4: Save adjacency matrix and buyer mappings
    sp.save_npz('UPU_adj_matrix.npz', adj_matrix)
    with open('buyer_to_idx.json', 'w') as f:
        json.dump({buyer: idx for idx, buyer in enumerate(unique_buyers)}, f)

    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    print(f"Total unique buyers: {len(unique_buyers)}")

    # Optional: Verify the matrix
    # verify_matrix(adj_matrix, unique_buyers, buyer_to_products)

if __name__ == '__main__':
    main()