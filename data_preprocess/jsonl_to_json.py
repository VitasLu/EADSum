import json

# 讀取JSONL檔案
with open('./datasets/cnndm/cnndm_test.jsonl', 'r', encoding='utf-8') as jsonl_file:
    json_list = [json.loads(line) for line in jsonl_file]  # 解析每一行

# 將資料寫入JSON檔案
with open('./datasets/cnndm/cnndm_test_transform.json', 'w', encoding='utf-8') as json_file:
    json.dump(json_list, json_file, ensure_ascii=False, indent=4)  # 格式化並寫入
