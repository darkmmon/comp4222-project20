import json
import numpy as np

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



def read_json_file(path = 'data_preprocess_matrix/Subscription_Boxes.jsonl'):
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
                
    return np.array(matrix, dtype = object)       #[rating, product_ID, user_ID, timestamp] * 16216 (buy record, not user)


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


# def create_UPU_matrix(buyer_n_products):
#     # Get the list of unique buyers
#     unique_buyers = list(buyer_n_products.keys())    #convert key in dictionary to list
#     # print(unique_buyers)
#     num_buyers = len(unique_buyers)

#     # Initialize an empty adjacency matrix
#     adjacency_matrix = np.zeros((num_buyers, num_buyers), dtype=int)

#     # buyer_to_idx = {buyer: idx for idx, buyer in enumerate(unique_buyers)}    #map buyers to integer 
#     # print(buyer_to_idx[unique_buyers[1]])

#     for i in range(num_buyers):
#         for j in range(i + 1, num_buyers):  # Avoid self-connections and redundant checks
#             buyer_i = unique_buyers[i]
#             buyer_j = unique_buyers[j]

#             # Check if buyer_i and buyer_j share any products
#             if buyer_n_products[buyer_i] & buyer_n_products[buyer_j]:  # Intersection of product sets
#                 adjacency_matrix[i, j] = 1
#                 adjacency_matrix[j, i] = 1  # Ensure symmetry

#     return adjacency_matrix, unique_buyers



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
    homo_matrix = upu_matrix | usv_matrix  # Element-wise OR to combine connections
    return homo_matrix


if __name__ == '__main__':
    matrix = read_json_file()

    buyers_assigned_products = assign_products_to_buyers(matrix)   #(total 15327 buyers (nodes))
    
    upu_matrix, unique_buyers = create_UPU_matrix(buyers_assigned_products)
    
    buyer_to_idx = create_buyer_to_idx(unique_buyers)
