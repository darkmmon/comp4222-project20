from read_json_file import read_json_file
from USV_pre.assign_ratings_to_users import assign_ratings_to_users
from USV_pre.create_adjacency_matrix import create_adjacency_matrix
from USV_pre.create_usv_matrix import create_usv_matrix

def main():
    
    # Step 1: Read the data from the JSON file
    matrix = read_json_file()

    # Step 2: Create the USV adjacency matrix
    product_with_users = assign_ratings_to_users(matrix)

    # Step 3: Print the USV adjacency matrix
    user_connections = create_usv_matrix(product_with_users)

    # Step 4: Get the list of all unique users   
    all_users = list(user_connections.keys())

    # Step 5: Create the adjacency matrix
    usv_matrix = create_adjacency_matrix(user_connections, all_users)
    
    # Step 6: Print the USV adjacency matrix
    print("USV Adjacency Matrix:")
    print(usv_matrix)




if __name__ == "__main__":
    main()