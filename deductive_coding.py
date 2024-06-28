import os
import pandas as pd
from src.vRA import RaLLM
from src.llm_coder import deductive_coding
from utils.krippendorff_alpha import krippendorff
from openai import OpenAI
import argparse

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', type = str, default = './data/data_example.csv')
    argparser.add_argument('--codebook', type = str, default = './data/codebook_example.csv')
    argparser.add_argument('--save', type = str, default = 'results/results_example.csv')
    argparser.add_argument('--mode', type = str, default = 'deductive_coding')
    argparser.add_argument('--codebook_format', type=str, default = 'codebook')
    argparser.add_argument('--context', type = bool,  default = False)
    argparser.add_argument('--number_of_example', type = int,  default = 5)
    argparser.add_argument('--voter', type = int,  default = 1)
    argparser.add_argument('--language', type = str, default = 'en')
    argparser.add_argument('--key', type = str, default = None)
    argparser.add_argument('--model', type = str, default = 'gpt-4o-2024-05-13')
    argparser.add_argument('--verification', type = bool,  default = False)
    argparser.add_argument('--batch', type = bool,  default = False)
    argparser.add_argument('--na_label', type = bool,  default = False)
    argparser.add_argument('--cot', type = bool,  default = False)
    argparser.add_argument('--base_url', type = str,  default = "")
    args = argparser.parse_args()

    if args.key:
        api_key = args.key
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    if args.mode == 'deductive_coding':
        data = pd.read_csv(args.input)
        codebook = pd.read_csv(args.codebook)
        if args.base_url:
            client = OpenAI(
                api_key = api_key,
                base_url = args.base_url,
            )
        else:
            client = OpenAI(
                api_key = api_key
            )
        results, code_set = deductive_coding(data, codebook, codebook_format = args.codebook_format, number_of_example =args.number_of_example, context = args.context, na_label = args.na_label, language = args.language, model = args.model,voter = args.voter, cot = args.cot, client = client, batch = args.batch)
        results.to_csv(args.save, encoding="utf_8_sig", index=False)
        #Calculate the Cohen's Kappa and Krippendorff's Alpha
        if args.verification:
            print(results['code'])
            print("Cohen's Kappa: %.3f" %RaLLM.cohens_kappa_measure(data['code'].astype(str), data['results']))
            print("Krippendorff's Alpha: %.3f" %RaLLM.krippendorff_alpha_measure(data['code'].astype(str), data['results'],code_set))

if __name__=="__main__":
    main()
