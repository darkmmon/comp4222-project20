import numpy as np

def create_usv_matrix(matrix):
    # Get the number of records
    num_records = len(matrix)
    
    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((num_records, num_records), dtype=int)

    # Define the time window for one week in milliseconds
    one_week_in_millis = 7 * 24 * 60 * 60 * 1000  # 7 days in milliseconds

    # Loop over all pairs of records
    for i in range(num_records):
        for j in range(i + 1, num_records):  # Avoid redundant checks and self-connections
            user_i = matrix[i][4]  # User ID for record i
            user_j = matrix[j][4]  # User ID for record j
            
            rating_i = matrix[i][0]  # Rating for record i
            rating_j = matrix[j][0]  # Rating for record j
            
            timestamp_i = matrix[i][5]  # Timestamp for record i
            timestamp_j = matrix[j][5]  # Timestamp for record j
            
            # Check if the ratings are the same and the reviews are within one week
            if rating_i == rating_j and abs(timestamp_i - timestamp_j) <= one_week_in_millis:
                # Connect user i and user j
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1  # Ensure symmetry

    return adjacency_matrix