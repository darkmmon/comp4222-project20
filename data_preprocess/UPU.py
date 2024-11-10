# goal: if two node buying the same product -> mean two node are connected -> index 1 in the adjacency matrix  /  if not, 0

from read_json_file import read_json_file
from UPU_pre.assign_products_to_buyers import assign_products_to_buyers
from UPU_pre.create_UPU_matrix import create_UPU_matrix
import numpy as np



def main():
    matrix = read_json_file()

    buyer_to_products = assign_products_to_buyers(matrix)   #(total 15327 buyers (nodes))

    adjacency_matrix, unique_buyers = create_UPU_matrix(buyer_to_products)

    print(adjacency_matrix.shape)
    print(len(unique_buyers))     #1128 node(buyers)

if __name__ == '__main__':
    main() 