# vRA: Virtual Research Assistant
vRA is a Python Toolkit designed to assist social science discovery with AI. The current version supports deductive coding, coding qualitative data into quantitative ones based on a pre-developed codebook.


## Installation

1. Clone this repository:
```
git clone https://github.com/isle-dev/vRA.git
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Set up an environment variable named OPENAI_API_KEY with your OpenAI API key. Alternatively, you can provide the API key as a command-line argument when running the main script

## Deductive Coding
Deductive coding is a process of coding qualitative data into quantitative ones based on a pre-developed codebook. The current implementation of deductive coding is based on the [Supporting Qualitative Analysis with Large Language Models: Combining Codebook with GPT-3 for Deductive Coding](https://dl.acm.org/doi/abs/10.1145/3581754.3584136). Please refer to our paper for further details. 

```
@inproceedings{xiao2023supporting,
  title={Supporting Qualitative Analysis with Large Language Models: Combining Codebook with GPT-3 for Deductive Coding},
  author={Xiao, Ziang and Yuan, Xingdi and Liao, Q Vera and Abdelghani, Rania and Oudeyer, Pierre-Yves},
  booktitle={Companion Proceedings of the 28th International Conference on Intelligent User Interfaces},
  pages={75--78},
  year={2023}
}
```

### IMPORTANT NOTE
We are currently running a meta-analysis study to understand the capability and limitation of this tool in deductive coding. If you are interested in contribute to this study, please contact Ziang Xiao at ziang dot xiao at jhu dot edu.

```
This LLM-based method is sensitive to codebook design and coding context. Please read our paper before using this tool. 
We  **strongly recommend** verifying the results with experts before using them for further analysis. 
```


### Current Implementation
The current implementation supports GPT-4. We will update the code to support future models. 

Language support: English, French, Chinese.

### Data Format
There are two CSV files required for deductive coding: the data CSV file and the codebook CSV file. 

The data CSV file contains the following columns.
```
data: The text data to be coded.
context: The context for the text data.
code: the assigned code [only needed for verification].
```

The codebook CSV file contains the codebook for deductive coding. The codebook CSV file should have the following columns:
```
code: The code for deductive coding.
description: The description of the code.
example_i: The ith example for the code.
context_i: The context for the ith example.
```

### Usage
Run the main script llm_coder.py using the following command:
```
python main.py --input data/data_example.csv --codebook data/codebook_example.csv --save results/results_example.csv --mode deductive_coding  
```

### Arguments
You can customize the behavior of the script by modifying the command-line arguments:
```
--input: Path to the input CSV file containing the text data.
--codebook: Path to the codebook CSV file containing the codebook for deductive coding.
--save: Path to the output CSV file where the results will be saved.
--mode: The mode in which to run the script. Choose 'deductive_coding'.
--codebook_format: The format of the codebook. Default is 'codebook'. Alternatively you can choose 'example', for more details see the original paper.
--context: Whether to include context for the deductive coding (1 = include, 0 = do not include). Default is 0.
--number_of_example: The number of examples to include in the codebook prompt. Default is 5.
--voter: The number of geneartions for each data point. If n > 1, the final code is an aggreation of mutiple generations by majority vote. Default is 1.
--language: The language of the input data and codebook (en, fr, or ch). Default is 'en'.
--key: Your OpenAI API key. If not provided, the script will attempt to use the OPENAI_API_KEY environment variable.
--model: The name of the GPT model to use (e.g., 'gpt-4-0613', 'text-davinci-003'). Default is 'gpt-4-0613'.
--verification: Whether to calculate Cohen's Kappa and Krippendorff's Alpha for inter-coder reliability (1 = calculate, 0 = do not calculate). Default is 1. Note: code column is required in the input CSV file.
--batch_size: The batch size for saving the coding progress. Default is 100 (reuslts will be saved for everyon 100 data points).
--na_label: Whether to include an "NA" label (1 = include, 0 = do not include). Default is 0.
```

## Contact
If you have any questions, please contact [Ziang Xiao](https://www.ziangxiao.com/) at ziang dot xiao at jhu dot edu.

