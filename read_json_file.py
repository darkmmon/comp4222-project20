import json
import numpy as np

def read_json_file(path = 'Subscription_Boxes.jsonl'):
    matrix = []

    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                json_obj = json.loads(line)
                
                rating = json_obj.get('rating')
                length_of_title = len(json_obj.get('title'))
                timestamp = json_obj.get('timestamp')
                helpful_vote = json_obj.get('helpful_vote')
                product_ID = json_obj.get('asin')
                verified_purchase = json_obj.get('verified_purchase')
            

                matrix.append((rating, length_of_title, timestamp, helpful_vote, product_ID, verified_purchase))
                
    return np.array(matrix, dtype = object)

# matrix = read_json_file()
# print(matrix)