## Exploring Large Language Models’ Diagnostic Capabilities for Condition Prediction and Therapy Recommendations Using Clinical Case Reports

Project Overview: This study evaluates the capability of the LLM models in diagnosing diseases and recommending interventional & pharmpharmacological therapies from Metadata Acquired from Clinical Case Reports (MACCRs) 
    
##### Data Preprocessing
For data preprocessing, run   
``` shell
python data_preprocess.py
```

##### Prediction
For prediction of y attr using GPT models, run   
``` shell
python pred_gpt.py --model gpt_model_name --sample_mode sample_mode --sample_size k --y_attr prediction_attribute
```
Example:
``` shell
python pred_gpt.py --model gpt-4o-mini --sample_mode random_k --sample_size 5 --y_attr diagnosis
```

For prediction of y attr using huggingface models, run   
``` shell
python pred_huggingface.py --model huggingface_model_name --sample_mode sample_mode --sample_size k --y_attr prediction_attribute
```
Example:
``` shell
python pred_huggingface.py --model starmpcc/Asclepius-Mistral-7B-v0.3 --sample_mode random_k --sample_size 5 --y_attr diagnosis
```

For prediction of y attr using ollama models, run   
``` shell
python pred_ollama.py --model ollama_model_name --sample_mode sample_mode --sample_size k --y_attr prediction_attribute
```
Example:
``` shell
python pred_ollama.py --model llama3 --sample_mode random_k --sample_size 5 --y_attr diagnosis
```

To evaluate prediction made by a model, run   
``` shell
python eval.py --model eval_model_name --filepath result_csv_filepath --gt groundtruth_col_name --pred predicted_col_name
```
Example:
``` shell
python eval.py --model gpt-4o-mini --filepath pred_diagnosis_zeroshot_0_results.csv --gt diagnosis --pred diagnosis_pred_by_llama3
```
