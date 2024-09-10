import pandas as pd
import argparse
from tqdm import tqdm

from metrics import *
from client import create_client

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"], required=True)
parser.add_argument('--filepath', type=str, required=True)
parser.add_argument('--gt', type=str, required=True)
parser.add_argument('--pred', type=str, required=True)
args = parser.parse_args()

def main():
    fp = args.filepath
    df = pd.read_csv(fp)
    client = create_client()
    grades = []
    total_eval_token_usage = 0
    for index, (diag_gt, diag_pred) in tqdm(enumerate(zip(df[args.gt], df[args.pred]))):
        if args.gt == 'diagnosis':
            grade, grade_token_usage = grade_diagnosis(diag_gt, diag_pred, client, args.model)
        elif args.gt == 'interventional_therapy':
            grade, grade_token_usage =  grade_interventional_prediction(diag_gt, diag_pred, client, args.model)
        elif args.gt == 'pharmacological_therapy':
            grade, grade_token_usage =  grade_pharmacological_prediction(diag_gt, diag_pred, client, args.model)
        total_eval_token_usage += grade_token_usage
        grades.append(grade) 
    print(f"Total tokens used for case evaluation query: {total_eval_token_usage}")
    column_name = f'{args.pred}_graded_by_{args.model}'
    df[column_name] = grades
    column_name = f'{args.pred}_cosine_similarity'
    cosine_score = cal_similarities(df[args.gt], df[args.pred], client, model='small')
    df[column_name] = cosine_score
    df.to_csv(fp, index=False)

if __name__ == "__main__":
    main()