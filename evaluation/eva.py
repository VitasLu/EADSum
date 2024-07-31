# -!- coding: utf-8 -!-
import json
import os
from metric import BatchEvaluation
import argparse


def batch_evalution(dataset, start_id, end_id, bs_true):
    # in_file = os.path.join("../data", dataset+"_element_aware.json")
    
    model = "pegasus-large_FT100"
    # 可選: "bart-large_FT100", "bart-base_FT100", "pegasus-large_FT100", "t5-base", "t5-large", "longt5-base", "longt5-large"
    
    in_file = os.path.join(f"./output/cnndm_{model}.json")
    with open(in_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    eva_ori_lt5 = BatchEvaluation()  # (original ref. summary) vs. (longT5 summary)
    eva_ori_dis = BatchEvaluation()  # (original ref. summary) vs. (distill_longT5 summary)
    eva_ori_gpt = BatchEvaluation()  # (original ref. summary) vs. (GPT-3.5 std. summary)
    eva_ori_cot = BatchEvaluation()  # (original ref. summary) vs. (GPT-3 cot summary)
    
    eva_ele_lt5 = BatchEvaluation()  # (element-aware ref. summary) vs. (longT5 summary)
    eva_ele_dis = BatchEvaluation()  # (element-aware ref. summary) vs. (GPT-3 std. summary)
    eva_ele_gpt = BatchEvaluation()  # (element-aware ref. summary) vs. (GPT-3.5 std. summary)
    eva_ele_cot = BatchEvaluation()  # (original ref. summary) vs. (GPT-3 cot summary)

    for i in range(start_id, end_id + 1):
        ori_ref = data[i]["original_summary"]
        ele_ref = data[i]["element-aware_summary"]
        # std_pred = data[i]["gpt3.5_summary"]
        # cot_pred = data[i]["gpt3.5_cot_summary"]
        # dislongT5_pred = data[i]["T5_DT20"]
        longT5_pred = data[i][f"{model}"]

        # if ori_ref == "" or ele_ref == "" or std_pred == "" or cot_pred == "" or dislongT5_pred == "":
        #     continue

        eva_ori_lt5.set_text(ori_ref, longT5_pred)
        eva_ori_lt5.get_rouge_score()
        if bs_true: eva_ori_lt5.get_bs_score()

        # eva_ori_dis.set_text(ori_ref, dislongT5_pred)
        # eva_ori_dis.get_rouge_score()
        # if bs_true: eva_ori_dis.get_bs_score()

        # eva_ori_gpt.set_text(ori_ref, std_pred)
        # eva_ori_gpt.get_rouge_score()
        # if bs_true: eva_ori_gpt.get_bs_score()

        # eva_ori_cot.set_text(ori_ref, cot_pred)
        # eva_ori_cot.get_rouge_score()
        # if bs_true: eva_ori_cot.get_bs_score()

        eva_ele_lt5.set_text(ele_ref, longT5_pred)
        eva_ele_lt5.get_rouge_score()
        if bs_true: eva_ele_lt5.get_bs_score()

        # eva_ele_dis.set_text(ele_ref, dislongT5_pred)
        # eva_ele_dis.get_rouge_score()
        # if bs_true: eva_ele_dis.get_bs_score()

        # eva_ele_gpt.set_text(ele_ref, std_pred)
        # eva_ele_gpt.get_rouge_score()
        # if bs_true: eva_ele_gpt.get_bs_score()

        # eva_ele_cot.set_text(ele_ref, cot_pred)
        # eva_ele_cot.get_rouge_score()
        # if bs_true: eva_ele_cot.get_bs_score()
    
    print(f"original ref. summary vs. {model} summary:\n"
        f"batch size:{eva_ori_lt5.call_time_rs}\n"
        f"r1: {eva_ori_lt5.total_r1 / eva_ori_lt5.call_time_rs}\n"
        f"r2: {eva_ori_lt5.total_r2 / eva_ori_lt5.call_time_rs}\n"
        f"rl: {eva_ori_lt5.total_rl / eva_ori_lt5.call_time_rs}\n"
        f"bs: {eva_ori_lt5.total_bs / eva_ori_lt5.call_time_rs}\n")
            

    # print(f"original ref. summary vs. T5_DT20 summary:\n"
    #     f"batch size:{eva_ori_dis.call_time_rs}\n"
    #     f"r1: {eva_ori_dis.total_r1 / eva_ori_dis.call_time_rs}\n"
    #     f"r2: {eva_ori_dis.total_r2 / eva_ori_dis.call_time_rs}\n"
    #     f"rl: {eva_ori_dis.total_rl / eva_ori_dis.call_time_rs}\n"
    #     f"bs: {eva_ori_dis.total_bs / eva_ori_dis.call_time_rs}\n")

    # print(f"original ref. summary vs. GPT-3.5 std. summary:\n"
    #     f"batch size:{eva_ori_gpt.call_time_rs}\n"
    #     f"r1: {eva_ori_gpt.total_r1 / eva_ori_gpt.call_time_rs}\n"
    #     f"r2: {eva_ori_gpt.total_r2 / eva_ori_gpt.call_time_rs}\n"
    #     f"rl: {eva_ori_gpt.total_rl / eva_ori_gpt.call_time_rs}\n"
    #     f"bs: {eva_ori_gpt.total_bs / eva_ori_gpt.call_time_rs}\n")

    # print(f"original ref. summary vs. GPT-3.5 cot summary:\n"
    #     f"batch size:{eva_ori_cot.call_time_rs}\n"
    #     f"r1: {eva_ori_cot.total_r1 / eva_ori_cot.call_time_rs}\n"
    #     f"r2: {eva_ori_cot.total_r2 / eva_ori_cot.call_time_rs}\n"
    #     f"rl: {eva_ori_cot.total_rl / eva_ori_cot.call_time_rs}\n"
    #     f"bs: {eva_ori_cot.total_bs / eva_ori_cot.call_time_rs}\n")

    print(f"element-aware ref. summary vs. {model} summary:\n"
        f"batch size:{eva_ele_lt5.call_time_rs}\n"
        f"r1: {eva_ele_lt5.total_r1 / eva_ele_lt5.call_time_rs}\n"
        f"r2: {eva_ele_lt5.total_r2 / eva_ele_lt5.call_time_rs}\n"
        f"rl: {eva_ele_lt5.total_rl / eva_ele_lt5.call_time_rs}\n"
        f"bs: {eva_ele_lt5.total_bs / eva_ele_lt5.call_time_rs}\n")

    # print(f"element-aware ref. summary vs. T5_DT20 summary:\n"
    #     f"batch size:{eva_ele_dis.call_time_rs}\n"
    #     f"r1: {eva_ele_dis.total_r1 / eva_ele_dis.call_time_rs}\n"
    #     f"r2: {eva_ele_dis.total_r2 / eva_ele_dis.call_time_rs}\n"
    #     f"rl: {eva_ele_dis.total_rl / eva_ele_dis.call_time_rs}\n"
    #     f"bs: {eva_ele_dis.total_bs / eva_ele_dis.call_time_rs}\n")

    # print(f"element-aware ref. summary vs. GPT-3.5 std. summary:\n"
    #     f"batch size:{eva_ele_gpt.call_time_rs}\n"
    #     f"r1: {eva_ele_gpt.total_r1 / eva_ele_gpt.call_time_rs}\n"
    #     f"r2: {eva_ele_gpt.total_r2 / eva_ele_gpt.call_time_rs}\n"
    #     f"rl: {eva_ele_gpt.total_rl / eva_ele_gpt.call_time_rs}\n"
    #     f"bs: {eva_ele_gpt.total_bs / eva_ele_gpt.call_time_rs}\n")

    # print(f"element-aware ref. summary vs. GPT-3.5 cot summary:\n"
    #     f"batch size:{eva_ele_cot.call_time_rs}\n"
    #     f"r1: {eva_ele_cot.total_r1 / eva_ele_cot.call_time_rs}\n"
    #     f"r2: {eva_ele_cot.total_r2 / eva_ele_cot.call_time_rs}\n"
    #     f"rl: {eva_ele_cot.total_rl / eva_ele_cot.call_time_rs}\n"
    #     f"bs: {eva_ele_cot.total_bs / eva_ele_cot.call_time_rs}\n")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--dataset", type=str, default="cnndm",
                        choices=["cnndm", "xsum"], help="dataset source")
    parser.add_argument("--start_id", type=int, default="0")
    parser.add_argument("--end_id", type=int, default="199")
    parser.add_argument("--bs_true", type=bool, default=True)
    args = parser.parse_args()
    #args.end_id = args.start_id
    batch_evalution(dataset=args.dataset, start_id=args.start_id, end_id=args.end_id, bs_true=args.bs_true)
