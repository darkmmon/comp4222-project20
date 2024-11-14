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
