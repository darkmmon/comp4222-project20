
def assign_products_to_buyers(matrix):
    buyer_with_products = {}

    for row in matrix:
        buyer_id = row[4]        # Buyer ID
        product_id = row[3]      # Product ID (asin)

        # If the buyer is not in the dictionary, initialize an empty set for their products
        if buyer_id not in buyer_with_products:
            buyer_with_products[buyer_id] = set()

        # Add the product to the buyer's set of products
        buyer_with_products[buyer_id].add(product_id)
    
    # print(len(buyer_with_products))

    return buyer_with_products

