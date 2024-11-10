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