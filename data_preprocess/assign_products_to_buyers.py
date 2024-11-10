# assign product to buyers in dictionary from


def assign_products_to_buyers(matrix):
    buyer_to_products = {}

    for row in matrix:
        buyer_id = row[3]        # Buyer ID
        product_id = row[6]      # Product ID (asin)

        # If the buyer is not in the dictionary, initialize an empty set for their products
        if buyer_id not in buyer_to_products:
            buyer_to_products[buyer_id] = set()

        # Add the product to the buyer's set of products
        buyer_to_products[buyer_id].add(product_id)

    return buyer_to_products

