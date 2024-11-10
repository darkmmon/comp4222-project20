import json



def read_json_file(path = 'Subscription_Boxes.jsonl'):
    matrix = []

    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                json_obj = json.loads(line)
                
                rating = json_obj.get('rating')
                helpful_vote = json_obj.get('helpful_vote')
                verified_purchase = json_obj.get('verified_purchase')

                matrix.append((rating, helpful_vote, verified_purchase))

