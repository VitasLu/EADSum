import json

# 讀取JSON檔案
with open('./datasets/cnndm/train/cnndm_train_78706.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 處理每個項目，移除src中的換行符
for item in data:
    item['original_summary'] = item['original_summary'].replace('\n', '')
    item['element-aware'] = item['element-aware'].replace('\n', '')

# 將處理後的數據寫回JSON檔案
with open('./datasets/cnndm/train/cnndm_train_78706.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=2)

print("處理完成，結果已保存到 output.json")