#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
import nltk.data
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import sklearn.metrics.pairwise

from itertools import combinations

import xml.etree.ElementTree as ET

import re    
import numpy as np
import networkx as nx
import spacy
import os
import glob
import string
import json
import pickle

###############################################################################
#
# GLOBAL
#
###############################################################################

originalDS = 'Dataset_original/'
asTextDS = 'Dataset_as_txt/'
preprocDS = 'Dataset_preprocessed/'
goldenSet = 'Keyphrases_golden_set/'

#The following is creating folders to store keyphrases and pickles, one for each approach.
if not os.path.exists('Keyphrases_experimental_simple'):
    os.makedirs('Keyphrases_experimental_simple')
exprSetSimple = 'Keyphrases_experimental_simple/'

if not os.path.exists('Keyphrases_experimental_LP_OC'):
    os.makedirs('Keyphrases_experimental_LP_OC')
exprSetLP_OC = 'Keyphrases_experimental_LP_OC/'

if not os.path.exists('Keyphrases_experimental_LP_JAC'):
    os.makedirs('Keyphrases_experimental_LP_JAC')
exprSetLP_JAC = 'Keyphrases_experimental_LP_JAC/'

if not os.path.exists('Keyphrases_experimental_IDF_OC'):
    os.makedirs('Keyphrases_experimental_IDF_OC')
exprSetIDF_OC = 'Keyphrases_experimental_IDF_OC/'

if not os.path.exists('Keyphrases_experimental_IDF_JAC'):
    os.makedirs('Keyphrases_experimental_IDF_JAC')
exprSetIDF_JAC = 'Keyphrases_experimental_IDF_JAC/'

if not os.path.exists('PR_Simple'):
    os.makedirs('PR_Simple')
PR_Simple = 'PR_Simple/'

if not os.path.exists('PR_LP_OC'):
    os.makedirs('PR_LP_OC')
PR_LP_OC = 'PR_LP_OC/'

if not os.path.exists('PR_LP_JAC'):
    os.makedirs('PR_LP_JAC')
PR_LP_JAC = 'PR_LP_JAC/'

if not os.path.exists('PR_IDF_OC'):
    os.makedirs('PR_IDF_OC')
PR_IDF_OC = 'PR_IDF_OC/'

if not os.path.exists('PR_IDF_JAC'):
    os.makedirs('PR_IDF_JAC')
PR_IDF_JAC = 'PR_IDF_JAC/'

#word embeddings model, not being used, see report.
#emb = spacy.load('en_core_web_lg')

stop_words = set(stopwords.words('english'))

#types of weights
#EMB = "emb"
OCC = "occ"
JAC = "jac"

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
        
        mav = average_precision(gs, exp)
        
    return mav

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
    #not removing punctuation yet because countvectorizer will ignore it generating ngrams.
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
    
    #returns a list of candidates and a list where each candidate has its term-freq associated
    return result_ngrams, result_with_count



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

###############################################################################
#
# PRIORS
#
###############################################################################
#LP prior: uses the lenght and position of the candidate
def compute_prior_lp(candidates, sentences):
    n_sentences = len(sentences)
    result = dict()
    distribution = []
    
    for idx, candidate in enumerate(candidates):
        for i in range(n_sentences):
            if candidate in sentences[i]:
                if candidate not in result:
                    #Value will be given by the 1st sentence the cand is found, weighted by its lenght
                    result[candidate] = "temp"
                    distribution.append((n_sentences - i)*len(candidate))
        #candidate not in any of the sentences:
        #EXAMPLE: sentence: "topcoder hackerrank , managed". countvectorized will create 3gram: "topcoder hackrrank managed"
        #A candidate that was previously separated by punctiation won't probably be a keyword. Therefore we give a prob of 0.
        if candidate not in result:
            result[candidate] = "temp"
            distribution.append(0)            
                
    for i in range(len(distribution)):
        val = distribution[i]/sum(distribution)
        result[candidates[i]] = val

    return result

#TFIDF prior: uses the TFIDF value weighted by the number of words of the candidate
def compute_prior_idf(candidates, sentences, idf):
    n_sentences = len(sentences)
    result = dict()
    distribution = []
        
    for candidate in candidates:
        if(candidate[1] in idf):
            score = candidate[0] * idf[candidate[1]] * len(candidate[1].split())
            result[candidate[1]] = "temp"
            distribution.append(score)
        else: break
    
    for i in range(len(distribution)):
        val = distribution[i]/sum(distribution)
        result[candidates[i][1]] = val
    
    return result
                

###############################################################################
#
# EDGE WEIGHTS
#
###############################################################################

def compute_weights_embeddings(candidate1, candidate2):
    #when producing vectors for sentences, spacy takes the average of the vectores of the words
    c1_t = emb(candidate1)
    c2_t = emb(candidate2)

    return round((1 - c1_t.similarity(c2_t))*10, 1)


def compute_weights_occurencies(candidate1, candidate2, sentences):
    if candidate1 != candidate2:
        count = 0
        for sent in sentences:
            if candidate1 in sent and candidate2 in sent:
                count = count + 1
        
        #the bigger the count, the smaller the weight
        return round(1/count, 2)*10
    
    else:
        count = 0
        for sent in sentences:
            if sent.count(candidate1) >= 2:
                count = count + 1       
                
        return round(1/count, 2)*10
    
    
def compute_weights_jaccard(candidate1, candidate2):
    #returns jaccard distance.
    #we want to make the algorithm converge to unigrams of important words and bi/trigrams including those words
    label1 = set(candidate1.split())
    label2 = set(candidate2.split())
    return nltk.jaccard_distance(label1, label2)    
    

###############################################################################
#
# HELPERS
#
###############################################################################

def add_graph_nodes(graph, candidates):
    for c in candidates:
        graph.add_node(c)
    return graph
        
#generates unique pairs of candidates [(a,b) == (b,a)]. Adds the pair as an edge 
#if both elements appear in one sentence.
def add_graph_edges(graph, candidates, sentences, weights):
    
    pairs = list(combinations(candidates, 2))
    for pair in pairs:
        for sentence in sentences:
            if pair[0] in sentence and pair[1] in sentence:
                if weights == None:
                    graph.add_edge(pair[0], pair[1])
                elif weights == OCC:
                    graph.add_edge(pair[0], pair[1], weight=compute_weights_occurencies(pair[0], pair[1], sentences))
                #elif weights == EMB:
                    #graph.add_edge(pair[0], pair[1], weight=compute_weights_embeddings(pair[0], pair[1]))
                elif weights == JAC:
                    graph.add_edge(pair[0], pair[1], weight=compute_weights_jaccard(pair[0], pair[1]))
                break    
    
    #for the self loops
    #for candidate in candidates:
        #for sentence in sentences:
            #if sentence.count(candidate) >= 2:
                #if weights == None:
                    #graph.add_edge(candidate, candidate)
                #elif weights == OCC:
                    #graph.add_edge(candidate, candidate, weight=compute_weights_occurencies(candidate, candidate, sentences))
                ##elif weights == EMB:
                    ##graph.add_edge(candidate, candidate, weight=compute_weights_embeddings(candidate, candidate))
                #elif weights == JAC:
                    #graph.add_edge(candidate, candidate, weight=compute_weights_jaccard(candidate, candidate))
                #break    
    
    return graph

#given the pagerank score for the candidates, writes to files the top n      
def write_keyphrases(pr, n, filename, t):
    keyphrases = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:n]
    open(t + os.path.basename(filename), 'w', encoding = 'utf-8').close
    with open(t + os.path.basename(filename), 'a+', encoding = 'utf-8') as out:        
        for i in keyphrases:
            out.write(i[0] + '\n')    

 
#given the pagerank score for the candidates, prints (stdout) the top n    
def print_keyphrases(pr, n):
    for i in range(n):
        print(sorted(pr.items(), key=lambda x: x[1], reverse=True)[i]) 
 
 
def save_pr(pr, filename, t):
    with open(t + os.path.basename(filename), 'wb') as handle:
        pickle.dump(pr, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        
###############################################################################
#
# PageRank: Adapting weights and prior node distribution
#
###############################################################################        
        
#simple approach: no weights, no prior.        
def simple_pagerank():
    for preprocessed in glob.glob(preprocDS + '*.txt'):
        print(preprocessed)
        candidates = ngrams(preprocessed, 1, 3)[0]
        sentences = get_sentences(preprocessed)
        graph = nx.Graph()
        graph = add_graph_nodes(graph, candidates)
        graph = add_graph_edges(graph, candidates, sentences, None)     
        pr = nx.pagerank(graph, max_iter = 50)
        save_pr(pr, preprocessed, PR_Simple)
        write_keyphrases(pr, 5, preprocessed, exprSetSimple)

    
#using the LP prior
def lp_pagerank(w):
    keyphrases_folder = ""
    pickle_folder = ""
    
    if w == OCC:
        pickle_folder = PR_LP_OC
        keyphrases_folder = exprSetLP_OC
    elif w == JAC:
        pickle_folder = PR_LP_JAC
        keyphrases_folder = exprSetLP_JAC      
    
    for preprocessed in glob.glob(preprocDS + '*.txt'):
        print(preprocessed)
        candidates = ngrams(preprocessed, 1, 3)[0]
        sentences = get_sentences(preprocessed)
        prior = compute_prior_lp(candidates, sentences)
        candidates = prior.keys()
        graph = nx.Graph()
        graph = add_graph_nodes(graph, candidates)
        graph = add_graph_edges(graph, candidates, sentences, w)     
        pr = nx.pagerank(graph, max_iter=50, personalization=prior)
        save_pr(pr, preprocessed, pickle_folder)
        write_keyphrases(pr, 5, preprocessed, keyphrases_folder)
    
#using the IDF prior
def idf_pagerank(w):
    docs = files_to_list(preprocDS)
    all_docs_idf = idf(docs, 1, 3)
    
    keyphrases_folder = ""
    pickle_folder = ""
    
    if w == OCC:
        pickle_folder = PR_IDF_OC
        keyphrases_folder = exprSetIDF_OC
    elif w == JAC:
        pickle_folder = PR_IDF_JAC
        keyphrases_folder = exprSetIDF_JAC  
    #emb wont be calculated
    
    for preprocessed in glob.glob(preprocDS + '*.txt'):
        print(preprocessed)
        candidates, candidates_with_tf = ngrams(preprocessed, 1, 3)
        sentences = get_sentences(preprocessed)
        prior = compute_prior_idf(candidates_with_tf, sentences, all_docs_idf)
        candidates = prior.keys()
        graph = nx.Graph()
        graph = add_graph_nodes(graph, candidates)
        graph = add_graph_edges(graph, candidates, sentences, w)     
        pr = nx.pagerank(graph, max_iter=50, personalization=prior)
        save_pr(pr, preprocessed, pickle_folder)
        write_keyphrases(pr, 5, preprocessed, keyphrases_folder)


###############################################################################
#
# MAIN
#
###############################################################################

#xml and json were parsed using functions from part1: from_xml_to_txt and parse_json.
#docs are already preprocessed, but the used function was preproc_all_txt()

#run the multiple methods

simple_pagerank()

idf_pagerank(JAC)

lp_pagerank(JAC)

lp_pagerank(OCC)

idf_pagerank(OCC)

#compute the metrics
total_files = 0
mav_sum_simple = 0
mav_sum_lp_occ = 0
mav_sum_idf_occ = 0
mav_sum_lp_jac = 0
mav_sum_idf_jac = 0


for file in glob.glob(goldenSet + '*.txt'):
    av = metrics(file, exprSetSimple + os.path.basename(file))
    mav_sum_simple += av
    
    av = metrics(file, exprSetIDF_OC + os.path.basename(file))
    mav_sum_idf_occ += av
    
    av = metrics(file, exprSetLP_OC + os.path.basename(file))
    mav_sum_lp_occ += av    
    
    av = metrics(file, exprSetIDF_JAC + os.path.basename(file))
    mav_sum_idf_jac += av
    
    av = metrics(file, exprSetLP_JAC + os.path.basename(file))
    mav_sum_lp_jac += av    
    
    total_files += 1
    

print("MAV Simple: " + str(mav_sum_simple/total_files))
print("MAV IDF_JAC: " + str(mav_sum_idf_jac/total_files))
print("MAV IDF_OC: " + str(mav_sum_idf_occ/total_files))
print("MAV LP_OC: " + str(mav_sum_lp_occ/total_files))
print("MAV LP_JAC: " + str(mav_sum_lp_jac/total_files))    
print("done")