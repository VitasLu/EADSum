import json

# 讀取JSON檔案
with open('./datasets/cnndm/train/cnndm_train_21834_GPToutput_json.json', 'r', encoding='utf-8') as json_file:
    json_data = json.load(json_file)
    # print(json_data)

# 將資料寫入JSONL檔案
with open('./datasets/cnndm/train/cnndm_train.json', 'w', encoding='utf-8') as jsonl_file:
    for item in json_data['cnndm']:
        jsonl_file.write(json.dumps(item, ensure_ascii=False) + '\n')