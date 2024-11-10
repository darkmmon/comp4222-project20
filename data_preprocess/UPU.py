from read_json_file import read_json_file
from assign_products_to_buyers import assign_products_to_buyers
from create_UPU_matrix import create_UPU_matrix

# goal: if two node buying the same product -> mean two node are connected -> index 1 in the adjacency matrix  /  if not, 0


def main():
    matrix = read_json_file()

    buyer_to_products = assign_products_to_buyers(matrix)

    adjacency_matrix, unique_buyers = create_UPU_matrix(buyer_to_products)

    print(adjacency_matrix.shape)
    print(len(unique_buyers))

if __name__ == '__main__':
    main() 