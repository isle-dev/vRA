import re
from collections import Counter

def majority_vote(output):
    '''
    This function takes a list called output and returns the most common element
    in the list. If there is a tie between elements, the function returns one of
    the tied elements that appears first in the list.
    '''
    formatted_output = []
    for i in output:
        pattern = r'\{(.*?)\}'
        matches = re.findall(pattern, i)
        if matches:
            formatted_output.append(matches[0])
        else:
            formatted_output.append(i)
    x = Counter(formatted_output)
    return x.most_common(1)[0][0]