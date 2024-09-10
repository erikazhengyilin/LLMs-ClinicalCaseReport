import json
import random
import pandas as pd
import argparse
from tqdm import tqdm
import os

from client import create_client
from metrics import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"], required=True)
parser.add_argument('--sample_mode', type=str, choices=["random_k", "top_k", "zeroshot"], required=True)
parser.add_argument('--sample_size', type=int, required=True)
parser.add_argument('--y_attr', type=str, choices=["diagnosis", "pharmacological_therapy", "interventional_therapy"], required=True)
args = parser.parse_args()

def load_dataset_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def format_case(case):
    attrs_cols = ['Life Style', 'Family History',
       'Social History', 'Medical/Surgical History', 'Signs and Symptoms',
       'Comorbidities',
       'Laboratory Values', 
       'Pathology','Age', 'Gender']

    descriptions = "; ".join(f"{col}: {case.get(col, 'NA') if case.get(col) is not None else 'NA'}" for col in attrs_cols)
    diagnosis = case.get('Diagnosis', 'NA') if case.get('Diagnosis') is not None else 'NA'
    cr_number = case.get('CR Number', 'NA') if case.get('CR Number') is not None else 'NA'
    interventional = case.get('Interventional Therapy', 'NA') if case.get('Interventional Therapy') is not None else 'NA'
    pharmacological = case.get('Pharmacological Therapy', 'NA') if case.get('Pharmacological Therapy') is not None else 'NA'
 
    return cr_number, descriptions, diagnosis, interventional, pharmacological

def top_k_similar_case(case, dataset, k):
    case_embedding = np.array(case["case_embedding"]).reshape(1, -1)
    train_embeddings = np.array([train_case["case_embedding"] for train_case in dataset]).reshape(len(dataset), -1)
    similarities = cosine_similarity(case_embedding, train_embeddings).flatten()
    topk_idx = similarities.argsort()[-k:]
    topk = [dataset[idx] for idx in topk_idx]
    return topk

def create_fewshot_prompt(examples):
    prompt = ""
    for example in examples:
        cr_number, descriptions, diagnosis, interventional, pharmacological = format_case(example)
        prompt += f"Descriptions: {descriptions}\nDiagnosis: {diagnosis}\nInterventional Therapy: {interventional} \
            \nPharmacological Therapy: {pharmacological}\n\n"
    return prompt

def pred_fewshot(fewshot_prompt, case_description, client):
    prompt = f"Here are some examples,\n {fewshot_prompt} Now for Input: {case_description}\n, what is the suggested potential {args.y_attr}? \
        Please provide the potential {args.y_attr}, without any style or formatting."
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": f"You are a highly knowledgeable medical expert with expertise in diagnosing a \
             variety of medical conditions. You are presented with a patient's clinical details and are asked to provide \
             a potential {args.y_attr}"},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content, response.usage.total_tokens 

def pred_zeroshot(case_description, client):
    prompt = f"For a clinical case like this: {case_description}\n, what is the suggested potential {args.y_attr}? \
        Please provide the potential {args.y_attr}, without any style or formatting."
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": f"You are a highly knowledgeable medical expert with expertise in diagnosing a \
             variety of medical conditions. You are presented with a patient's clinical details and are asked to provide \
             a potential {args.y_attr}"},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content, response.usage.total_tokens 

def initialize_results_csv(dataset, results_fp):
    if os.path.exists(results_fp):
        results_df = pd.read_csv(results_fp)
    else:
        results_df = pd.DataFrame({
            'cr_number': dataset['CR Number'],
            'diagnosis': dataset['Diagnosis'],
            'pharmacological_therapy': dataset['Pharmacological Therapy'],
            'interventional_therapy': dataset['Interventional Therapy']
        })
        results_df.to_csv(results_fp, index=False)

    return results_df

def main():
    if args.y_attr == 'diagnosis':
        dataset = load_dataset_json('dataset/maccrs.json')
    elif args.y_attr == 'interventional_therapy':
        dataset = load_dataset_json('dataset/maccrs_intervthera.json')
    elif args.y_attr == 'pharmacological_therapy':
        dataset = load_dataset_json('dataset/maccrs_pharmthera.json')

    results_fp = f'pred_{args.y_attr}_{args.sample_mode}_{args.sample_size}_results.csv'
    results = initialize_results_csv(pd.DataFrame(dataset), results_fp)

    client = create_client()

    total_pred_token_usage = 0

    for i, case in tqdm(enumerate(dataset)):
        cr_number, descriptions, diag_gt, interventional, pharmacological = format_case(case)
        if args.sample_mode == 'random_k':
            k = args.sample_size
            train_sample = random.sample(dataset, k)
            fewshot_prompt = create_fewshot_prompt(train_sample)
            pred, pred_token_usage = pred_fewshot(fewshot_prompt, descriptions, client)
        elif args.sample_mode == 'top_k':
            k = args.sample_size
            topk_cases = top_k_similar_case(case, dataset, k)
            fewshot_prompt = create_fewshot_prompt(topk_cases)
            pred, pred_token_usage = pred_fewshot(fewshot_prompt, descriptions, client)
        elif args.sample_mode == 'zeroshot':
            pred, pred_token_usage = pred_zeroshot(descriptions, client)

        total_pred_token_usage += pred_token_usage

        matching_row_index = results[results['cr_number'] == cr_number].index
        if not matching_row_index.empty:
            results.loc[matching_row_index, f"{args.y_attr}_pred_by_{args.model}"] = pred
        else:
            print("No matching CR Number found. ", cr_number)

    print(f"Total tokens used for case prediction query: {total_pred_token_usage}")
    results.to_csv(results_fp, index=False)

if __name__ == "__main__":
    main()