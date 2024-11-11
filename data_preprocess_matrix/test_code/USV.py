from read_json_file import read_json_file
from USV_pre.assign_ratings_to_users import assign_ratings_to_users
from USV_pre.create_adjacency_matrix import create_adjacency_matrix
from USV_pre.create_usv_matrix import create_usv_matrix
import numpy as np

def is_matrix_all_zero(matrix):
    """
    This function checks if the given matrix is all zeros.
    
    Args:
    - matrix (np.ndarray): A NumPy matrix (2D array).
    
    Returns:
    - bool: True if all elements in the matrix are zero, False otherwise.
    """
    # Check if all elements in the matrix are zero
    return np.all(matrix == 0)

def main():
    # Step 1: Read the data from the JSON file
    matrix = read_json_file()
    
    # Step 2: Group users by product and get unique users
    product_with_users, unique_users = assign_ratings_to_users(matrix)
    
    # Step 3: Create user connections based on same rating and timestamps within 7 days
    user_connections = create_usv_matrix(product_with_users)
    
    # Step 4: Use the unique users collected from the dataset
    all_users = list(unique_users)
    
    # Step 5: Create the adjacency matrix
    usv_matrix = create_adjacency_matrix(user_connections, all_users)
    
    if is_matrix_all_zero(usv_matrix):
        print("The matrix is all zeros.")
    else:
        print("The matrix is not all zeros.")
    
    # # Step 6: Print the USV adjacency matrix dimensions
    # print("USV Adjacency Matrix Shape:", usv_matrix.shape)
    # print("USV Adjacency Matrix:\n", usv_matrix[4])

if __name__ == "__main__":
    main()
    
    
import json
import numpy as np

def read_json_file(path = 'data_preprocess/Subscription_Boxes.jsonl'):
    matrix = []

    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                json_obj = json.loads(line)
                
                rating = json_obj.get('rating')
                length_of_title = len(json_obj.get('title'))
                length_of_text = len(json_obj.get('text'))
                asin = json_obj.get("asin")
                user_id = json_obj.get('user_id')
                timestamp = json_obj.get('timestamp')
                helpful_vote = json_obj.get('helpful_vote')
                product_ID = json_obj.get('asin')
                verified_purchase = json_obj.get('verified_purchase')
                

                matrix.append((rating, length_of_title, length_of_text, asin, user_id, timestamp, helpful_vote, verified_purchase))
                
    return np.array(matrix, dtype = object)       #[rating, length of title, length of text, user_ID, timestamp, helpful_vote, product_ID, verified_purchase] * 16216 (buy record, not user)


if __name__ == '__main__':
    matrix = read_json_file()
    print(matrix[1])


#{"rating": 1.0, "title": "USELESS", "text": "Absolutely useless nonsense and a complete waste of money. Kitty didn't like any of the items", "images": [], "asin": "B07G584SHG", "parent_asin": "B09WC47S3V", "user_id": "AEMJ2EG5ODOCYUTI54NBXZHDJGSQ", "timestamp": 1602133857705, "helpful_vote": 2, "verified_purchase": true}

    




def assign_ratings_to_users(matrix):
    product_with_users = {}
    unique_users = set()  # To store unique user IDs

    for row in matrix:
        user_id = row[4]        # User ID
        product_id = row[3]     # Product ID (asin)
        rating = row[0]         # Rating
        timestamp = row[5]      # Timestamp
        
        unique_users.add(user_id)

        # If the product is not in the dictionary, initialize an empty list for its users
        if product_id not in product_with_users:
            product_with_users[product_id] = []

        # Add the user, rating, and timestamp to the product's list of users
        product_with_users[product_id].append((user_id, rating, timestamp))
    
    return product_with_users, unique_users



import numpy as np

def create_adjacency_matrix(user_connections, all_users):
    # Get the total number of unique users
    num_users = len(all_users)

    # Create a mapping from user_id to an index in the adjacency matrix
    user_to_idx = {user_id: idx for idx, user_id in enumerate(all_users)}

    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((num_users, num_users), dtype=int)

    # Fill the adjacency matrix based on user connections
    for user_id, connections in user_connections.items():
        for connected_user in connections:
            i = user_to_idx[user_id]
            j = user_to_idx[connected_user]
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1  # Ensure symmetry

    return adjacency_matrix

def create_usv_matrix(product_with_users):
    # Create an empty dictionary to store user connections
    user_connections = {}

    # Define the time window for one week in milliseconds (7 days)
    one_week_in_millis = 7 * 24 * 60 * 60 * 1000  # 7 days in milliseconds

    # Iterate over each product and its list of users
    for product_id, users in product_with_users.items():
        # Compare each pair of users for the current product
        for i in range(len(users)):
            for j in range(i + 1, len(users)):  # Avoid redundant checks and self-connections
                user_i, rating_i, timestamp_i = users[i]
                user_j, rating_j, timestamp_j = users[j]

                # Check if the ratings are the same and the reviews are within one week
                if rating_i == rating_j and abs(timestamp_i - timestamp_j) <= one_week_in_millis:
                    # Connect user_i and user_j
                    if user_i not in user_connections:
                        user_connections[user_i] = set()
                    if user_j not in user_connections:
                        user_connections[user_j] = set()
                    
                    user_connections[user_i].add(user_j)
                    user_connections[user_j].add(user_i)  # Ensure symmetry

    return user_connections