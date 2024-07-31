import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

checkpoint = "./baseline/T5-base/DT20/checkpoint-30000"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# load the dataset
with open('./datasets/cnndm/test/cnndm_test_200.json') as fs:
    original_dataset =  json.load(fs)

for data in original_dataset:
    text = f'{data["src"]}'
    # print("原文:", text)

    input_text = "summarize: " + text  
    input_tokens = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    summary_ids_dis = model.generate(input_tokens, max_length=256, num_beams=32, early_stopping=True)
    summary_text_dis = tokenizer.decode(summary_ids_dis[0], skip_special_tokens=True)
    data["t5-base_DT20_ck30000"] = summary_text_dis

    print("No.", data["id"])
    print("摘要:")
    print(summary_text_dis)

with open('./output/cnndm_t5-base_DT20_ck30000.json', "w") as f:
    json.dump(original_dataset, f, indent=4)
    print("Save to cnndm_t5-base_DT20_ck30000.json")


