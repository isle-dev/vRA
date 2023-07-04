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
import argparse
import tqdm
import time

def deductive_coding(args):
    """
    This example function demonstrates how to use the RaLLM package for deductive coding of qualitative data.

    The function reads data and a codebook from CSV files, generates the codebook prompt, and then uses the
    RaLLM package to obtain codes for each data point. The obtained codes are stored in a new column in the
    original DataFrame. Finally, the function calculates Cohen's Kappa or Krippendorff's Alpha to assess the
    inter-coder reliability.

    Returns:
    - DataFrame: A pandas DataFrame containing the original data and a new column with the obtained codes.
    """
    # Read data and codebook from CSV files
    data = pd.read_csv(args.input)
    codebook = pd.read_csv(args.codebook)
    # Generate the codebook prompt from the codebook
    codebook_prompt, code_set = RaLLM.codebook2prompt(codebook, format = args.codebook_format, num_of_examples = args.number_of_example, language = args.language, has_context = args.context)
    if args.na_label:
        code_set.append("NA")
    # Define the identity modifier and context description   
    if args.language == 'fr':
        meta_prompt = open('prompts/meta_prompt_fr.txt').read()
    elif args.language == 'ch':
        meta_prompt = open('prompts/meta_prompt_ch.txt').read()
    else:
        meta_prompt = open('prompts/meta_prompt_eng.txt').read()
    
    meta_prompt = meta_prompt.replace('{{CODE_SET}}', str(code_set))

    # Iterate through each row of the data
    results = []
    idx = 0
    for index, row in tqdm.tqdm(data.iterrows(), position=0,total=data.shape[0]):
        # Generate the final prompt
        prompt = RaLLM.prompt_writer(str(row['data']), str(row['context']), codebook_prompt, code_set, meta_prompt, args.na_label, args.language)
        # Obtain the code using the coder function from the RaLLM package
        if args.model == 'text-davinci-003':
            response = RaLLM.coder(prompt, engine = args.model)
            code = response["choices"][0]["text"].strip()
        else:
            response = RaLLM.coder(prompt, engine = args.model, voter = args.voter)
            code_voters= [response['choices'][i]['message']['content'] for i in range(len(response['choices']))]
            code = majority_vote(code_voters).strip()
        # Add the obtained code to the dataset
        results.append(code)
        idx += 1
        if idx%args.batch_size == 0:
            results = RaLLM.code_clean(results,code_set)
            data['result'] = pd.Series(results)
            csv_idx = args.save.index('.csv')
            file_name = args.save[:csv_idx] + '_'+str(idx)+ args.save[csv_idx:]
            data.to_csv(file_name, encoding="utf_8_sig", index=False)

    #NOTE: Please double check/post processing the results before beofre calculating the inter-rater reliability. Some codes maybe slightly different than the codebook.
    results = RaLLM.code_clean(results,code_set)
    data['result'] = pd.Series(results)
    data.to_csv(args.save, encoding="utf_8_sig", index=False)
    #Calculate the Cohen's Kappa and Krippendorff's Alpha
    if args.verification:
        print(data['code'])
        print("Cohen's Kappa: %.3f" %RaLLM.cohens_kappa_measure(data['code'].astype(str), data['result']))
        print("Krippendorff's Alpha: %.3f" %RaLLM.krippendorff_alpha_measure(data['code'].astype(str), data['result'],code_set))

    return data

@retry(
    reraise=True,
    stop=stop_after_attempt(1000),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(retry_if_exception_type(openai.error.Timeout)
        | retry_if_exception_type(openai.error.APIError)
        | retry_if_exception_type(openai.error.APIConnectionError)
        | retry_if_exception_type(openai.error.RateLimitError)),
)
def scale_single_taking():
    """
    This example function demonstrates how to use the RaLLM package to fill a series of Likert scale questions
    by obtaining scale values for each item individually.

    The function reads Likert scale questions from a CSV file, constructs a series of items, and then uses the
    RaLLM package to obtain scale values for each item separately. The obtained results are stored in a new
    column in the original DataFrame.

    Returns:
    - DataFrame: A pandas DataFrame containing the original data and a new column with the obtained scale values.
    """
    # Read the data from the CSV file
    data = pd.read_csv('./data/scale_example.csv')
    # Construct a series of items using the item_constructor function from the RaLLM package
    items = RaLLM.item_constructor(data, 50)
    # Define an identity modifier and the main prompt template
    identity_modifier = 'Please respond to the following items as if you are a [OCCUPATION]. '  
    main_prompt_template = "Answer each item on a 5-point scale:1 = Strongly disagree 2 = Disagree 3 = Neither agree nor disagree 4 = Agree 5 = Strongly agree. Fill the [BLANK] with a numerical answer." 
    # Combine the identity modifier and the main prompt template
    compelte_prompt = identity_modifier+main_prompt_template 
    # Use the scale_taker function from the RaLLM package to obtain scale values for each item individually
    results = [RaLLM.scale_taker(compelte_prompt,item) for item in items]
    # Add the obtained results as a new column in the original DataFrame 
    data['result'] = pd.Series(results)
    return data

@retry(
    reraise=True,
    stop=stop_after_attempt(1000),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(retry_if_exception_type(openai.error.Timeout)
        | retry_if_exception_type(openai.error.APIError)
        | retry_if_exception_type(openai.error.APIConnectionError)
        | retry_if_exception_type(openai.error.RateLimitError)),
)
def scale_seq_taking():
    """
    This example function demonstrates how to use the RaLLM package to fill a series of Likert scale questions in a single conversation session with gpt3.5 (chatgpt) or gpt4.

    The function reads Likert scale questions from a CSV file, constructs a series of items, and then uses the
    RaLLM package to obtain scale values for each item. The obtained results are stored in a new column in
    the original DataFrame.

    Returns:
    - DataFrame: A pandas DataFrame containing the original data and a new column with the obtained scale values.
    """
    # Read the data from the CSV file
    data = pd.read_csv('./data/scale_example.csv')
    # Construct a series of items using the item_constructor function from the RaLLM package
    items = RaLLM.item_constructor(data, 50)
    # Define an identity modifier and the main prompt template
    identity_modifier = 'Please respond to the following items as if you are a [OCCUPATION]. '  
    main_prompt_template = "Answer each item on a 5-point scale:1 = Strongly disagree 2 = Disagree 3 = Neither agree nor disagree 4 = Agree 5 = Strongly agree. Fill the [BLANK] with a numerical answer." 
    
    # Combine the identity modifier and the main prompt template
    compelte_prompt = identity_modifier+main_prompt_template
    
    # Use the scale_taker_seq function from the RaLLM package to obtain scale values for each item
    results = RaLLM.scale_taker_seq(compelte_prompt ,items)
    # Add the obtained results as a new column in the original DataFrame
    data['result'] = pd.Series(results)
    return data

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', type = str, default = './data/data_example.csv')
    argparser.add_argument('--codebook', type = str, default = './data/codebook_example.csv')
    argparser.add_argument('--save', type = str, default = 'results/results_example.csv')
    argparser.add_argument('--mode', type = str, default = 'deductive_coding')
    argparser.add_argument('--codebook_format', type=str, default = 'codebook')
    argparser.add_argument('--context', type = int,  default = 0)
    argparser.add_argument('--number_of_example', type = int,  default = 5)
    argparser.add_argument('--voter', type = int,  default = 1)
    argparser.add_argument('--language', type = str, default = 'en')
    argparser.add_argument('--key', type = str, default = None)
    argparser.add_argument('--model', type = str, default = 'gpt-4-0613')
    argparser.add_argument('--verification', type = int,  default = 0)
    argparser.add_argument('--batch_size', type = int,  default = 100)
    argparser.add_argument('--na_label', type = int,  default = 0)
    args = argparser.parse_args()

    if args.key:
        openai.api_key = args.key
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")

    if args.mode == 'deductive_coding':
        results = deductive_coding(args)

if __name__=="__main__":
    main()
