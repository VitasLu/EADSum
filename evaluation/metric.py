# -!- coding: utf-8 -!-
import json
import openai
import argparse
from rouge import Rouge
from bert_score import score
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
bart_model = BartForConditionalGeneration.from_pretrained(model_name)

def rouge_score(ref, pred):
    rouge = Rouge()
    rs = rouge.get_scores(pred, ref)
    rouge1 = rs[0]["rouge-1"]["f"] * 100
    rouge2 = rs[0]["rouge-2"]["f"] * 100
    rougel = rs[0]["rouge-l"]["f"] * 100
    return rouge1, rouge2, rougel


def bs_score(ref, pred):
    _, _, F1 = score([pred], [ref], lang="en", verbose=True)
    bs = F1.mean()
    return bs

def bart_score(ref, pred):
    input_ids = tokenizer(ref, return_tensors="pt").input_ids
    candidate_ids = tokenizer(pred, return_tensors="pt").input_ids

    with torch.no_grad():
        outputs = bart_model(input_ids=input_ids, labels=candidate_ids)
        loss = outputs.loss

    bart_score_value = -loss.item()
    return bart_score_value

class BatchEvaluation:
    def __init__(self, total_r1=0, total_r2=0, total_rl=0, total_bs=0, total_bart=0,
                 call_time_rs=0, call_time_bs=0):
        self.ref = ""
        # print("self.ref: ", self.ref)
        self.pred = ""
        # print("self.pred: ", self.pred)

        self.total_r1 = total_r1
        self.total_r2 = total_r2
        self.total_rl = total_rl
        self.total_bs = total_bs
        self.total_bart = total_bart
        self.call_time_rs = call_time_rs
        self.call_time_bs = call_time_bs
        self.call_time_bart = call_time_bart

    def set_text(self, ref, pred):
        self.ref = ref
        self.pred = pred
        return self

    def get_rouge_score(self):
        r1, r2, rl = rouge_score(self.ref, self.pred)
        self.total_r1 += r1
        self.total_r2 += r2
        self.total_rl += rl
        self.call_time_rs += 1

    def get_bs_score(self):
        bs = bs_score(self.ref, self.pred)
        self.total_bs += bs
        self.call_time_bs += 1

    def get_bart_score(self):
        bart = bart_score(self.ref, self.pred)
        self.total_bart += bart
        self.call_time_bart += 1