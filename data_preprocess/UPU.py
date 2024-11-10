# goal: if two node buying the same product -> mean two node are connected -> index 1 in the adjacency matrix  /  if not, 0

import numpy as np

from read_json_file import read_json_file
from UPU_pre.assign_products_to_buyers import assign_products_to_buyers
from UPU_pre.create_UPU_matrix import create_UPU_matrix


def main():
    matrix = read_json_file()

    buyer_to_products = assign_products_to_buyers(matrix)   #(total 15327 buyers (nodes))

    upu_matrix, unique_buyers = create_UPU_matrix(buyer_to_products)

    print(upu_matrix.shape)
    print(len(unique_buyers))     #1128 node(buyers)

if __name__ == '__main__':
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


import numpy as np

def create_UPU_matrix(buyer_n_products):
    # Get the list of unique buyers
    unique_buyers = list(buyer_n_products.keys())    #convert key in dictionary to list
    # print(unique_buyers)
    num_buyers = len(unique_buyers)

    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((num_buyers, num_buyers), dtype=int)

    # buyer_to_idx = {buyer: idx for idx, buyer in enumerate(unique_buyers)}    #map buyers to integer 
    # print(buyer_to_idx[unique_buyers[1]])

    for i in range(num_buyers):
        for j in range(i + 1, num_buyers):  # Avoid self-connections and redundant checks
            buyer_i = unique_buyers[i]
            buyer_j = unique_buyers[j]

            # Check if buyer_i and buyer_j share any products
            if buyer_n_products[buyer_i] & buyer_n_products[buyer_j]:  # Intersection of product sets
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1  # Ensure symmetry

    return adjacency_matrix, unique_buyers