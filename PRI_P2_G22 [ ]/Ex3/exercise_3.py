#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from math import log

import glob
import pickle
import os

###############################################################################
#
# GLOBAL
#
###############################################################################

graphCS = 'Pickles/PR_IDF_OC/'
asTextDS = 'Dataset_as_txt/'
preprocDS = 'Dataset_preprocessed/'
goldenSet = 'Keyphrases_golden_set/'

#create this folder to store the results
if not os.path.exists('Keyphrases_experimental'):
    os.makedirs('Keyphrases_experimental')
exprSet = 'Keyphrases_experimental/'

stop_words = set(stopwords.words('english'))

###############################################################################
#
# METRICS
#
###############################################################################
def rates(gs, exp):
    tp = 0
    fp = 0

    for i in exp:
        if i in gs:
            tp += 1
        if i not in gs:
            fp += 1
            
    return tp, fp

def precision(tp, fp):
    return 0 if tp + fp == 0 else tp/(tp+fp)

def average_precision(golden, experimental):
    precisions = []
    for i in range(1,len(experimental)+1):
        if experimental[:i][-1] in golden:
            tp, fp = rates(golden, experimental[:i])
            precisions.append(precision(tp, fp))
            
    
    avp = sum(precisions)/min(len(golden), len(experimental))
    return avp

def metrics(golden, experimental):
    with open(golden, 'r', encoding = 'utf-8') as gs_file, open(experimental, 'r', encoding = 'utf-8') as exp_file:
        gs = []
        exp = []
        for line in gs_file:
            gs.append(line)
        for line in exp_file:
            exp.append(line)
        
        av = average_precision(gs, exp)
        
    return av

###############################################################################
#
# PREPROCESSING
#
###############################################################################

def preproc_all_txt():
    for filename in glob.glob(asTextDS + '*.txt'):
        noPath = os.path.basename(filename)
        out = preprocDS + noPath
        
        preprocess_file(filename, out)

def preprocess_file(docname, outfile_name):
    #given a file docname, preprocesses and saves it at file outfile_name
    with open(docname, 'r', encoding = 'utf-8') as file:
        outfile = open(outfile_name, 'w', encoding = 'utf-8')
    
        for line in file:
            print(preprocess_sentence(line), end='', file=outfile)
        outfile.close()
        
    return outfile_name


def preprocess_sentence(sentence):
    #preprocess a sentece: lowercase it, remove stopwords.
    #not removing punctuation yet because countvectorizer will ignore it while generating ngrams.
    #also final points are useful to split in sentences
    processed = sentence.lower()
    tokens = word_tokenize(processed)
    filtered_words = [w for w in tokens if not w in stop_words]
    return " ".join(filtered_words)

###############################################################################
#
# OTHER FUNCTIONS
#
###############################################################################
#given a file, get ngrams from range low to high.
def ngrams(docname, low, high):
    #given a file, get ngrams from range low to high.
    with open(docname, 'r', encoding = 'utf-8') as document:
        result_with_count = []
        result_ngrams = []
        
        c_vec = CountVectorizer(ngram_range=(low, high))
        ngrams = c_vec.fit_transform(document)
        vocab = c_vec.vocabulary_
        count_values = ngrams.toarray().sum(axis=0)
            
        for ngram in sorted([[count_values[i],k] for k,i in vocab.items()], reverse=True):
            result_with_count.append(ngram)
            result_ngrams.append(ngram[1])

    return result_ngrams, result_with_count


#aux function for bm25
def n_t(list1, list2):
    c=[]
    for i in range(0,len(list1)):
        n=0
        for j in range(0,len(list2)):
            if (list1[i] in list2[j]) == True:
                n=n+1
        c.append((n,list1[i]))
    return c  

#aux function for bm25
def frequency(list1, doc1):
    c=[]
    for i in range(0,len(list1)):
        c.append(doc1.count(list1[i]))
    return c

#aux function for bm25
def avgdl(list1):
    s=0
    for i in range(0,len(list1)):
        s = s+len(list1[i])
    total = s/len(list1)
    return total

def bm25(candidates, dataset, doc):    
    N = len(dataset)
    D = len(doc) #variation, see report
    k1 = 1.2
    b = 0.75
    avgdl1 = avgdl(dataset)
    frequency1 = frequency(candidates, doc)
    n_t1 = n_t(candidates, dataset)
    final = dict()
    for i in range(0,len(candidates)):
        score = log((N - n_t1[i][0] + 0.5)/(n_t1[i][0] + 0.5))*((frequency1[i]*(k1+1))/(frequency1[i]+(k1*(1-b+(b*(D/avgdl1))))))
        final[candidates[i]] = score
    return sorted(final.items(), key=lambda x: x[1], reverse=True)

#compute idf values for ngrams in a dataset represented as a list
def idf(dataset, low, high):
    vectorizer = TfidfVectorizer(ngram_range=(low, high))

    vectorizer.fit_transform(dataset)

    idf = vectorizer.idf_

    return dict(zip(vectorizer.get_feature_names(), idf))   

#returns list of sentences of a document
def get_sentences(docname):
    #using punkt tokenizer as it avoids many problmes
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    fp = open(docname)
    data = fp.read()
    return tokenizer.tokenize(data) 


def get_sentence_text(text):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return tokenizer.tokenize(text)

#returns a list where each entry is the text of a file in a given directory
def files_to_list(directory):
    result = []
    for filename in glob.glob(directory + '*.txt'):
        with open(filename, 'r', encoding='utf-8') as document:  
            string = document.read()
        result = result + [string]   
    return result


def write_keyphrases(score, n, filename):
    keyphrases = sorted(score.items(), key=lambda x: x[1], reverse=True)[:n]
    open(exprSet + os.path.basename(filename), 'w', encoding = 'utf-8').close
    with open(exprSet + os.path.basename(filename), 'a+', encoding = 'utf-8') as out:        
        for i in keyphrases:
            out.write(i[0] + '\n')

def print_keyphrases(score, n):
    for i in range(n):
        print(sorted(score.items(), key=lambda x: x[1], reverse=True)[i])
        
        
        
def asTxt(filename):
    with open(preprocDS + os.path.basename(filename), 'r', encoding='utf-8') as document:
        string = document.read()
    return string    
###############################################################################
#
# FEATURIZING
#
###############################################################################

#Returns an ordered list of lists, where each candidate has its correspondent weighted tfidf score
def weighted_tfidf_score(candidates, idf):
    w_tfidf = dict()
    for cand in candidates:
        w_tfidf[cand[1]] = cand[0] * idf[cand[1]] * len(cand[1].split())
        
    return sorted(w_tfidf.items(), key=lambda x: x[1], reverse=True)

#Returns an ordered list of lists, where each candidate has its correspondent tfidf score
def unweighted_tfidf_score(candidates, idf):
    w_tfidf = dict()
    for cand in candidates:
        w_tfidf[cand[1]] = cand[0] * idf[cand[1]]
        
    return sorted(w_tfidf.items(), key=lambda x: x[1], reverse=True)

#Returns an ordered list of lists, where each candidate has its correspondent pr score
def centrality_scores(filename):
    #Computing the scores again takes ALOT of time. so we used a pickle.
    with open(graphCS + os.path.basename(filename), 'rb') as handle:
        pr = pickle.load(handle)    
        
    return sorted(pr.items(), key=lambda x: x[1], reverse=True)    
    
#Returns an ordered list of lists, where each candidate has its correspondent position in doc
def position_in_document(filename, candidates):
    pos = dict()
    with open(preprocDS + os.path.basename(filename), 'r', encoding = 'utf-8') as file:
        text = file.read() 
    
    for cand in candidates:
        if(cand[1] in text):
            pos[cand[1]] = text.index(cand[1])
        else:
            pos[cand[1]] = len(text) + 1
        
    return sorted(pos.items(), key=lambda x: x[1], reverse=False)

#returns a dict: {'candX':[rank_wtfidf, rank_pr], ...}
def candidates_as_ranks(features, candidates):
    result = dict()
    #get ranks from the measures and compute result:
    for candidate in candidates:
        result[candidate[1]] = []
        for feature in features:
            for candidate_score in feature:
                if candidate_score[0] == candidate[1]:
                    rank = feature.index(candidate_score) +1 
                    result[candidate[1]].append(rank)    
                
    return result

###############################################################################
#
# RECIPROCAL RANK FUSION
#
###############################################################################

#receives a dictionary: {'candX':[rank_f1, rank_f2, ...], ...}
def RRFS(candidates_ranks):
    result = dict()

    for candidate_rank in candidates_ranks.items():
        result[candidate_rank[0]] = 0
        for rank in candidate_rank[1]:
            result[candidate_rank[0]] += (1/(50+rank))
            
    return result
        

###############################################################################
#
# MAIN
#
###############################################################################
#docs are already preprocessed, but the used function was preproc_all_txt()

#get all files to a list, compute the IDF (if using an IDF metric)
docs = files_to_list(preprocDS)
all_docs_idf = idf(docs, 1, 3)


for preprocessed in glob.glob(preprocDS + '*.txt'):
    print(preprocessed)        
    candidates, candidates_with_tf = ngrams(preprocessed, 1, 3)
    
    #all features: choose the ones to compute
    
    #bm25_score = bm25(candidates, docs, asTxt(preprocessed))
    #pos = position_in_document(preprocessed, candidates_with_tf)
    #unweighted_tfidf = unweighted_tfidf_score(candidates_with_tf, all_docs_idf)
    weighted_tfidf = weighted_tfidf_score(candidates_with_tf, all_docs_idf)
    pagerank = centrality_scores(preprocessed)
    
    #pass the chosen features: 
    features = (weighted_tfidf, pagerank)
    ranks = candidates_as_ranks(features, candidates_with_tf)
    scores = RRFS(ranks)
    write_keyphrases(scores, 5, preprocessed)


#compute metrics                                              
total_files = 0
mav_rrfs = 0

for file in glob.glob(goldenSet + '*.txt'):
    av = metrics(file, exprSet + os.path.basename(file))
    mav_rrfs += av
    
    total_files += 1

print("MAV RRFS: " + str(mav_rrfs/total_files))
print("done")


























