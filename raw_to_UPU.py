import numpy as np
from read_json_file import read_json_file

# goal: if two node buying the same product -> mean two node are connected -> index 1 in the adjacency matrix  /  if not, 0

def raw_to_UPU(raw_matrix = read_json_file()):
    product = raw_matrix[:, 4]
    
    unique_products = np.unique(product)    #find unique id in the array
    num_products = len(unique_products)
    # print(num_products)
    adjacency_matrix = np.zeros((num_products, num_products), dtype=int)
    
    product_to_idx = {product_id: idx for idx, product_id in enumerate(unique_products)}
    
    
    for i in range(num_products):
        for j in range(num_products):
            if i != j:  # No self-connections (optional)
                adjacency_matrix[i, j] = 1

    return adjacency_matrix, unique_products




print(raw_to_UPU())