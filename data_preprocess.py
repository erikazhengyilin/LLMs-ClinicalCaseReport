import pandas as pd
import numpy as np
import json
import os
import threading
from collections import OrderedDict
from sklearn.model_selection import train_test_split

from client import create_client, get_embedding_small

def convert_tsv_to_json(input_filepath, output_filepath):
    df = pd.read_csv(input_filepath, sep='\t')
    df.to_json(output_filepath, orient='records', indent=2)

def get_embeddings(filename, client):
    attrs_cols = ['Life Style', 'Family History',
       'Social History', 'Medical/Surgical History', 'Signs and Symptoms',
       'Comorbidities',
       'Laboratory Values', 
       'Pathology','Age', 'Gender']
        
    df = pd.read_json(filename)
    df = df[df['Diagnosis'].notna() & df['Diagnosis'].str.strip().astype(bool)]
    store_cols = [col for col in attrs_cols if col in df.columns]
    df['combined_text'] = df[store_cols].apply(lambda x: ' '.join(x.fillna('').astype(str)), axis=1)
    df['case_embedding'] = df['combined_text'].apply(
        lambda x: get_embedding_small(x, client).tolist() if x.strip() else None
    )
    df.drop(columns=['combined_text'], inplace=True)
    df.to_json(filename, orient='records', indent=2)

def main():
    input_filepath = 'dataset/MACCRs.tsv'
    output_basepath = 'dataset/maccrs.json'

    # convert to json
    convert_tsv_to_json(input_filepath, output_basepath)
    client = create_client()
    get_embeddings(output_basepath, client)

    # drop those that is NA for pharmacological therapy
    df = pd.read_json(output_basepath)
    df['Pharmacological Therapy'] = df['Pharmacological Therapy'].replace("", np.nan)
    df_filtered = df.dropna(subset=['Pharmacological Therapy'])
    df_filtered.to_json('dataset/maccrs_pharmthera.json', orient='records', indent=4)

    # drop those that is NA for interventional therapy
    df = pd.read_json(output_basepath)
    df['Interventional Therapy'] = df['Interventional Therapy'].replace("", np.nan)
    df_filtered = df.dropna(subset=['Interventional Therapy'])
    df_filtered.to_json('dataset/maccrs_intervthera.json', orient='records', indent=4)

if __name__ == "__main__":
    main()