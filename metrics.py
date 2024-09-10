from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import ollama
from tqdm import tqdm

from client import *

def cal_cosine_similarity(text1, text2, client, model='small'):
    if model == 'adav2':
        cos_similarity = cosine_similarity(get_embedding_ada(text1, client), \
                                           get_embedding_ada(text2, client))[0][0]
    elif model == 'small':
        cos_similarity = cosine_similarity(get_embedding_small(text1, client), \
                                           get_embedding_small(text2, client))[0][0]
    elif model == 'large':
        cos_similarity = cosine_similarity(get_embedding_large(text1, client), \
                                           get_embedding_large(text2, client))[0][0]
    elif model == 'bert':
        cos_similarity = cosine_similarity(get_embedding_bert(text1, client), \
                                           get_embedding_bert(text2, client))[0][0]
    return cos_similarity

def cal_similarities(diag_list1, diag_list2, client, model='small'):
    results = []
    for i, diag1 in tqdm(enumerate(diag_list1)):
        diag2 = diag_list2[i]
        cos_similarity = cal_cosine_similarity(diag1, diag2, client, model)
        results.append(cos_similarity)
    return results
 
def grade_diagnosis(text1, text2, client, eval_model):
    prompt = f"Please evaluate the accuracy of the predicted diagnosis '{text1}' compared to the ground truth diagnosis \
        '{text2}' on a scale of 1-5:\n1 - Very Inaccurate: The diagnosis is almost completely incorrect compared to the \
        ground truth. It misses key aspects of the condition, might identify an entirely different issue, \
        or lacks substantial clinical relevance. Any treatment based on this diagnosis could potentially harm the patient.\n \
        2 - Inaccurate: The diagnosis captures some elements related to the ground truth but is largely incorrect. \
        It may identify some symptoms or related conditions but misses the main diagnosis or significantly misunderstands the \
        condition. Treatment based on this diagnosis would likely be ineffective or only partially helpful.\n \
        3 - Partially Accurate: The diagnosis is somewhat correct, capturing a significant part of the ground truth but with notable \
        inaccuracies or omissions. It may identify the correct condition but miss important details, or it may correctly diagnose a related \
        issue but not the primary one. Treatment based on this diagnosis might be somewhat effective but would need substantial adjustment.\n \
        4 - Mostly Accurate: The diagnosis closely matches the ground truth with only minor inaccuracies or omissions. \
        It correctly identifies the condition and most of its key aspects, but may miss some subtle details. \
        Treatment based on this diagnosis would likely be effective but could be improved with further refinement.\n \
        5 - Highly Accurate: The diagnosis perfectly or almost perfectly matches the ground truth. \
        It accurately identifies the condition and all its key aspects, with no significant errors or omissions. \
        Treatment based on this diagnosis would be fully appropriate and effective. Please just return either one of these 5 phrases (Very Inaccurate, \
        Inaccurate, Partially Accurate, Mostly Accurate, Highly Accurate)"
    response = client.chat.completions.create(
        model=eval_model,
        messages=[
            {"role": "system", "content": "You are a medical assistant capable of assessing the accuracy of medical diagnoses. \
             Provide an evaluation based on the given ground truth and predicted diagnoses."},
            {"role": "user", "content": prompt}
        ],
    )
    grade = response.choices[0].message.content
    return grade, response.usage.total_tokens

def grade_pharmacological_prediction(prediction, ground_truth, client, eval_model):
    prompt = f"Please evaluate the accuracy of the predicted pharmacological therapy '{prediction}' compared to the ground truth therapy \
        '{ground_truth}' on a scale of 1-5:\n1 - Very Inaccurate: The therapy recommendation is almost completely incorrect compared to the \
        ground truth. It may involve prescribing entirely inappropriate medications, dosages, or treatment plans that could potentially harm the patient.\n \
        2 - Inaccurate: The therapy recommendation captures some elements related to the ground truth but is largely incorrect. \
        It might suggest some relevant medications or dosages but misses the main treatment strategy or significantly misunderstands the \
        condition. The recommended therapy would likely be ineffective or only partially helpful.\n \
        3 - Partially Accurate: The therapy recommendation is somewhat correct, capturing a significant part of the ground truth but with notable \
        inaccuracies or omissions. It might suggest the correct class of medications but at incorrect dosages, or it might recommend a therapy that is relevant \
        but not optimal for the primary condition. The treatment could be somewhat effective but would need substantial adjustment.\n \
        4 - Mostly Accurate: The therapy recommendation closely matches the ground truth with only minor inaccuracies or omissions. \
        It correctly identifies the necessary medications and dosages for the condition but may miss some subtle details or possible optimizations. \
        The recommended treatment would likely be effective but could be improved with further refinement.\n \
        5 - Highly Accurate: The therapy recommendation perfectly or almost perfectly matches the ground truth. \
        It accurately identifies the required medications, dosages, and treatment plans with no significant errors or omissions. \
        The recommended therapy would be fully appropriate and effective. Please just return either one of these 5 phrases (Very Inaccurate, \
        Inaccurate, Partially Accurate, Mostly Accurate, Highly Accurate)"
    response = client.chat.completions.create(
        model=eval_model,
        messages=[
            {"role": "system", "content": "You are a medical assistant capable of assessing the accuracy of pharmacological therapy recommendations. \
             Provide an evaluation based on the given ground truth and predicted therapies."},
            {"role": "user", "content": prompt}
        ],
    )
    grade = response.choices[0].message.content
    return grade, response.usage.total_tokens

def grade_interventional_prediction(prediction, ground_truth, client, eval_model):
    prompt = f"Please evaluate the accuracy of the predicted interventional therapy '{prediction}' compared to the ground truth therapy \
        '{ground_truth}' on a scale of 1-5:\n1 - Very Inaccurate: The therapy recommendation is almost completely incorrect compared to the \
        ground truth. It may involve prescribing entirely inappropriate medications, dosages, or treatment plans that could potentially harm the patient.\n \
        2 - Inaccurate: The therapy recommendation captures some elements related to the ground truth but is largely incorrect. \
        It might suggest some relevant medications or dosages but misses the main treatment strategy or significantly misunderstands the \
        condition. The recommended therapy would likely be ineffective or only partially helpful.\n \
        3 - Partially Accurate: The therapy recommendation is somewhat correct, capturing a significant part of the ground truth but with notable \
        inaccuracies or omissions. It might suggest the correct class of medications but at incorrect dosages, or it might recommend a therapy that is relevant \
        but not optimal for the primary condition. The treatment could be somewhat effective but would need substantial adjustment.\n \
        4 - Mostly Accurate: The therapy recommendation closely matches the ground truth with only minor inaccuracies or omissions. \
        It correctly identifies the necessary medications and dosages for the condition but may miss some subtle details or possible optimizations. \
        The recommended treatment would likely be effective but could be improved with further refinement.\n \
        5 - Highly Accurate: The therapy recommendation perfectly or almost perfectly matches the ground truth. \
        It accurately identifies the required medications, dosages, and treatment plans with no significant errors or omissions. \
        The recommended therapy would be fully appropriate and effective. Please just return either one of these 5 phrases (Very Inaccurate, \
        Inaccurate, Partially Accurate, Mostly Accurate, Highly Accurate)"
    response = client.chat.completions.create(
        model=eval_model,
        messages=[
            {"role": "system", "content": "You are a medical assistant capable of assessing the accuracy of interventional therapy recommendations. \
             Provide an evaluation based on the given ground truth and predicted therapies."},
            {"role": "user", "content": prompt}
        ],
    )
    grade = response.choices[0].message.content
    return grade, response.usage.total_tokens

def grade_diagnosis_huggingface(text1, text2, eval_model):
    prompt = f"Please evaluate the accuracy of the predicted diagnosis '{text1}' compared to the ground truth diagnosis \
        '{text2}' on a scale of 1-5:\n1 - Very Inaccurate: The diagnosis is almost completely incorrect compared to the \
        ground truth. It misses key aspects of the condition, might identify an entirely different issue, \
        or lacks substantial clinical relevance. Any treatment based on this diagnosis could potentially harm the patient.\n \
        2 - Inaccurate: The diagnosis captures some elements related to the ground truth but is largely incorrect. \
        It may identify some symptoms or related conditions but misses the main diagnosis or significantly misunderstands the \
        condition. Treatment based on this diagnosis would likely be ineffective or only partially helpful.\n \
        3 - Partially Accurate: The diagnosis is somewhat correct, capturing a significant part of the ground truth but with notable \
        inaccuracies or omissions. It may identify the correct condition but miss important details, or it may correctly diagnose a related \
        issue but not the primary one. Treatment based on this diagnosis might be somewhat effective but would need substantial adjustment.\n \
        4 - Mostly Accurate: The diagnosis closely matches the ground truth with only minor inaccuracies or omissions. \
        It correctly identifies the condition and most of its key aspects, but may miss some subtle details. \
        Treatment based on this diagnosis would likely be effective but could be improved with further refinement.\n \
        5 - Highly Accurate: The diagnosis perfectly or almost perfectly matches the ground truth. \
        It accurately identifies the condition and all its key aspects, with no significant errors or omissions. \
        Treatment based on this diagnosis would be fully appropriate and effective. Please just return either one of these 5 phrases (Very Inaccurate, \
        Inaccurate, Partially Accurate, Mostly Accurate, Highly Accurate)"
    response = ollama.chat(model=eval_model, messages=[
            {"role": "system", "content": "You are a medical assistant capable of assessing the accuracy of medical diagnoses. \
             Provide an evaluation based on the given ground truth and predicted diagnoses."},
            {"role": "user", "content": prompt}
        ])
    grade = response['message']['content']
    return grade