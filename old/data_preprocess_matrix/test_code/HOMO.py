import numpy as np

from read_json_file import read_json_file
from UPU_pre.assign_products_to_buyers import assign_products_to_buyers
from UPU_pre.create_UPU_matrix import create_UPU_matrix

from read_json_file import read_json_file
from USV_pre.assign_ratings_to_users import assign_ratings_to_users
from USV_pre.create_adjacency_matrix import create_adjacency_matrix
from USV_pre.create_usv_matrix import create_usv_matrix



def main():
    matrix = read_json_file()

    buyer_to_products = assign_products_to_buyers(matrix)   #(total 15327 buyers (nodes))

    upu_matrix, unique_buyers = create_UPU_matrix(buyer_to_products)

    matrix1 = read_json_file()
    
    # Step 2: Group users by product and get unique users
    product_with_users, unique_users = assign_ratings_to_users(matrix1)
    
    # Step 3: Create user connections based on same rating and timestamps within 7 days
    user_connections = create_usv_matrix(product_with_users)
    
    # Step 4: Use the unique users collected from the dataset
    all_users = list(unique_users)
    
    
    # Step 5: Create the adjacency matrix
    usv_matrix = create_adjacency_matrix(user_connections, all_users)
    
    HOMO = upu_matrix + usv_matrix
    
    print(unique_buyers[4399])
    print(all_users[4399])


if __name__ == '__main__':
    main() 