#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer

import nltk
import nltk.data
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

from itertools import combinations

import re    

import networkx as nx

###############################################################################
#
# GLOBAL
#
###############################################################################

document = 'original.txt'
preprocessed = 'preprocessed.txt'
stop_words = set(stopwords.words('english'))

###############################################################################
#
# PREPROCESSING
#
###############################################################################

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

def ngrams(docname, low, high):
    #given a file, get ngrams from range low to high. ignores punctuation
    with open(docname, 'r', encoding = 'utf-8') as document:
        result_ngrams = []
        
        c_vec = CountVectorizer(ngram_range=(low, high))
        ngrams = c_vec.fit_transform(document)
        vocab = c_vec.vocabulary_
        count_values = ngrams.toarray().sum(axis=0)
            
        for ngram in sorted([[count_values[i],k] for k,i in vocab.items()], reverse=True):
            result_ngrams.append(ngram[1])

    return result_ngrams

#opens a file a returns a list of its sentences
def sentences(docname):
    #using punkt tokenizer as it avoids many problmes ["Mr. John", lowercase beggining , ...]
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    fp = open(docname)
    data = fp.read()
    return tokenizer.tokenize(data)    

###############################################################################
#
# HELPERS
#
###############################################################################

def add_graph_nodes(graph, candidates):
    for c in candidates:
        graph.add_node(c)
    return graph
        
#creates unique pairs of candidates, adding them as edges if they occur.        
def add_graph_edges(graph, candidates, sentences):
    pairs = list(combinations(candidates, 2))
    for pair in pairs:
        for sentence in sentences:
            if pair[0] in sentence and pair[1] in sentence:
                graph.add_edge(pair[0], pair[1])
                break
            
    #self-loop
    #for candidate in candidates:
        #for sentence in sentences:
            #if sentence.count(candidate) >= 2:
                #graph.add_edge(candidate, candidate)
                #break
    
    return graph

#given the pagerank score for the candidates, prints (stdout) the top n    
def print_keyphrases(pr, n):
    for i in range(n):
        print(sorted(pr.items(), key=lambda x: x[1], reverse=True)[i])        

###############################################################################
#
# MAIN
#
###############################################################################

#preprocess the file, get candidates and generate sentences
preprocess_file(document, preprocessed)
candidates = ngrams(preprocessed, 1, 3)
sentences = sentences(preprocessed)

#build the graph: damping factor is default so that d = 15%; unweighted.
graph = nx.Graph()
graph = add_graph_nodes(graph, candidates)
graph = add_graph_edges(graph, candidates, sentences)     

pr = nx.pagerank(graph, max_iter = 50)
print_keyphrases(pr, 5)
        
#This shows that the graph is indeed unweighted (only edges, no weights):        
#print(graph.edges.data())
