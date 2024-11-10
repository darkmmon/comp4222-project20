from read_json_file import read_json_file
from create_usv_matrix import create_usv_matrix

def main():

    matrix = read_json_file()

    usv_matrix = create_usv_matrix(matrix)

    print("USV Adjacency Matrix:")
    print(usv_matrix)

if __name__ == "__main__":
    main()