import json

def process_json_file(file_path, output_file_path):
    # 讀取JSON文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 過濾掉包含 "element-aware": null 的字典
    filtered_data = [item for item in data if not ("element-aware" in item and item["element-aware"] is None)]
    
    # 計算剩餘字典的數量
    remaining_count = len(filtered_data)
    
    # 將處理後的數據寫回文件
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(filtered_data, file, ensure_ascii=False, indent=2)
    
    return remaining_count

# 使用示例
file_path = '../baseline-gpt4o/cnndm_gpt3.5_rationale_80000.json'  # 請替換為您的JSON文件路徑  # 請替換為您的JSON文件路徑
output_file_path = './datasets/cnndm/train/cnndm_train_80000.json'  # 請替換為輸出文件路徑  # 請替換為輸出文件路徑
remaining_dictionaries = process_json_file(file_path, output_file_path)

print(f"處理完成。剩餘 {remaining_dictionaries} 個字典。")