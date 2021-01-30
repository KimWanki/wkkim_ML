import json
from collections import Counter
import math
import re
import numpy as np

def test(raw):
    hangul=re.compile('[^가-힣]')
    word_list=' '.join(hangul.sub(' ',raw).split())
    return word_list

def preprocess(raw_string_list):
    string_list=raw_string_list.split('.')
    return [test(s) for s in string_list]


def total(file_name):
    with open(file_name,'rt',encoding='UTF8') as json_file:
        data=json.load(json_file)
        sentence_list=[]
        for item in data["document"]:
            for paragraph in item["paragraph"]:
                sentence_list += preprocess(paragraph["form"])
        sentence_list=[w for w in sentence_list if w]
        total_str=' '.join(sentence_list)
    return total_str

def num_prob(sentence,ngram):
    counts=Counter([sentence[i:i+ngram] for i in range(len(sentence)-ngram+1)])
    normalize=sum(counts.values())
    probability_list=dict((k, float(v)/normalize) for k,v in counts.items())
    return probability_list

def num_count(sentence,ngram):
    counts=Counter([sentence[i:i+ngram] for i in range(len(sentence)-ngram+1)])
    return counts

def entropy(sentence,ngram):
    return -sum(p*np.log2(p) for p in num_prob(sentence,ngram).values())


training_set = "NLRW1900000011.json"
test_set1= 'WARW1900003745.json'
test_set2= 'NIRW1900000020.json'
training_unigram_entropy=entropy(total(training_set),1)
training_bigram_entropy=entropy(total(training_set),2)
training_trigram_entropy=entropy(total(training_set),3)
test_set1_unigram_entropy=entropy(total(test_set1),1)
test_set1_bigram_entropy=entropy(total(test_set1),2)
test_set1_trigram_entropy=entropy(total(test_set1),3)
test_set2_unigram_entropy=entropy(total(test_set1),1)
test_set2_bigram_entropy=entropy(total(test_set1),2)
test_set2_trigram_entropy=entropy(total(test_set1),3)

aa=num_count(total(training_set),1)
for i in num_count(total(training_set),1).keys():
    if i not in num_count(total(test_set1),1).keys():
        aa[i]=1


# def cross_entropy(training, test):
#     return -sum(p*np.log2(p) for p in num_prob(sentence,ngram).values())


