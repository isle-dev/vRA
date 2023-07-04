import re
from collections import Counter

def majority_vote(output):
    '''
    This function takes a list called output and returns the most common element
    in the list. If there is a tie between elements, the function returns one of
    the tied elements that appears first in the list.
    '''
    x = Counter(output)
    return x.most_common(1)[0][0]