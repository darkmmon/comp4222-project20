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