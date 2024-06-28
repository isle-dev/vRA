import os
import openai
import pandas as pd
from src.vRA import RaLLM
from utils.krippendorff_alpha import krippendorff
from utils.utils import majority_vote
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import json
import csv
import re
import tqdm
import time

def deductive_coding(data, codebook, client, codebook_format = 'codebook', number_of_example = 5, context = True, na_label = False, language = 'eng', model = 'gpt-4o-2024-05-13',voter = '1', cot= False, batch = False):
    """
    This example function demonstrates how to use the RaLLM package for deductive coding of qualitative data.

    The function reads data and a codebook from CSV files, generates the codebook prompt, and then uses the
    RaLLM package to obtain codes for each data point. The obtained codes are stored in a new column in the
    original DataFrame. Finally, the function calculates Cohen's Kappa or Krippendorff's Alpha to assess the
    inter-coder reliability.

    Returns:
    - DataFrame: A pandas DataFrame containing the original data and a new column with the obtained codes.
    """
    # Generate the codebook prompt from the codebook
    RaLLM_client=RaLLM(client)
    codebook_prompt, code_set = RaLLM_client.codebook2prompt(codebook, format = codebook_format, num_of_examples = number_of_example, language = language, has_context = context)
    if na_label:
        code_set.append("NA")

    # Define the identity modifier and context description   
    if language == 'fr':
        meta_prompt = open('prompts/meta_prompt_fr.txt').read()
    elif language == 'ch':
        meta_prompt = open('prompts/meta_prompt_ch.txt').read()
    else:
        meta_prompt = open('prompts/meta_prompt_eng.txt').read()
    
    meta_prompt = meta_prompt.replace('{{CODE_SET}}', str(code_set))

    # Iterate through each row of the data
    results = []
    model_exp = []
    idx = 0
    for index, row in tqdm.tqdm(data.iterrows(), position=0,total=data.shape[0]):
        # Generate the final prompt
        prompt = RaLLM_client.prompt_writer(str(row['data']), str(row['context']), codebook_prompt, code_set, meta_prompt, na_label, language, cot)
        if batch:
            results.append({"custom_id": str(idx), "method": "POST", "url": "/v1/chat/completions", "body": {"model": model, "messages": [{"role": "user", "content": prompt}],"max_tokens": 1000}})
        else:
        # Obtain the code using the coder function from the RaLLM package
            response = RaLLM_client.coder(prompt, engine = model, voter = voter)
            try:
                code_voters= [response.choices[i].message.content for i in range(len(response.choices))]
                code = majority_vote(code_voters).strip()
            except:
                code = str(response)
            # Add the obtained code to the dataset
            results.append(code)
            if cot:
                model_exp.append(code)
        idx += 1
    
    if batch:
        return results, code_set
    if cot:
        data['model_exp'] = pd.Series(model_exp)

    results = RaLLM.code_clean(results,code_set)
    data['results'] = pd.Series(results)
    
    return data, code_set

