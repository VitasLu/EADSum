import json

# 讀取JSON檔案
with open('./datasets/cnndm/train/cnndm_train_287113.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 確保數據是一個列表
if not isinstance(data, list):
    raise ValueError("JSON檔案的頂層結構應該是一個列表")

# 取出前143556筆資料
extracted_data = data[30000:]

# 將提取的數據寫入新的JSON檔案
with open('./datasets/cnndm/train/cnndm_train_257113.json', 'w', encoding='utf-8') as file:
    json.dump(extracted_data, file, ensure_ascii=False, indent=4)

print(f"成功提取 {len(extracted_data)} 筆資料並保存到 'cnndm_train_257113.json'")