


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


