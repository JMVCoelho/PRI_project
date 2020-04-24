#!/usr/bin/python
# -*- coding:  utf-8 -*-
import requests
from bs4 import BeautifulSoup
import numpy as np
import urllib.request

import nltk
import nltk.data
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from math import log

import glob
import pickle
import os

from itertools import combinations
import networkx as nx

import json

import collections

import spacy


###############################################################################
#
# GLOBAL
#
###############################################################################

graphCS = 'Pickles_no_self_loops/PR_IDF_OC/'
if not os.path.exists('Dataset_as_txt'):
    os.makedirs('Dataset_as_txt')
asTextDS = 'Dataset_as_txt/'
if not os.path.exists('Dataset_preprocessed'):
    os.makedirs('Dataset_preprocessed')
preprocDS = 'Dataset_preprocessed/'
goldenSet = 'Keyphrases_golden_set/'

stop_words = set(stopwords.words('english'))

emb = spacy.load('en_core_web_lg')

EMB = 'emb'
###############################################################################
#
# SCRAPER
#
###############################################################################

#gets tittle and description from each item. commented code gets text from news url.
#need bs4 installed
def scraper():
    with open('Technology.xml', 'rb') as file:
        soup = BeautifulSoup(file, features="xml")
    items = soup.findAll('item')
    news_items = []
    for item in items:
        news_item = {}
        news_item['title'] = item.title.text
        news_item['description'] = item.description.text
        news_item['link'] = item.link.text
        news_items.append(news_item)
        files = ['file1.txt', 'file2.txt', 'file3.txt','file4.txt', 'file5.txt', 'file6.txt','file7.txt', 'file8.txt', 'file9.txt','file10.txt', 'file11.txt', 'file12.txt','file13.txt', 'file14.txt', 'file15.txt','file16.txt', 'file17.txt', 'file18.txt','file19.txt', 'file20.txt', 'file21.txt']
    for i in range(0,len(news_items)):
        text_file = open(asTextDS + files[i], "w")
        text_file.write(news_items[i]['title'] + '\n')
        text_file.write(news_items[i]['description'] + '\n')
        #urllib.request.urlretrieve(news_items[i]['link'])
        #url1 = news_items[i]['link']
        #r1 = requests.get(url1)
        #coverpage = r1.content
        #soup1 = BeautifulSoup(coverpage, 'lxml')
        #coverpage_news = soup1.find_all('p', class_='css-exrw3m evys1bk0')
        #final = ''
        #soup = BeautifulSoup(coverpage, 'lxml').find_all('p', class_='css-exrw3m evys1bk0')
        #for j in range(0,len(soup)):
            #final = final + soup[j].get_text()
        #text_file.write(final)
        text_file.close()

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
    with open(docname, 'r', encoding = 'latin1') as file:
        outfile = open(outfile_name, 'w', encoding = 'latin1')

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
    if sentence[-1] == '\n':
        return ' '.join(filtered_words) + ' '
    return ' '.join(filtered_words)

###############################################################################
#
# OTHER FUNCTIONS
#
###############################################################################

#given a file, get ngrams from range low to high.
def ngrams(docname, low, high):
    #given a file, get ngrams from range low to high.
    with open(docname, 'r', encoding = ' latin1') as document:
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
        with open(filename, 'r', encoding='latin1') as document:
            string = document.read()
        result = result + [string]
    return result

def value_for_file(filename, score, n):
    result = []
    with open(asTextDS + os.path.basename(filename), 'r', encoding = ' latin1') as file:
        result.append(file.read())
        
    for i in range(n):
        result.append(sorted(score.items(), key=lambda x: x[1], reverse=True)[i])
    
    return result
    
def asTxt(filename):
    with open(preprocDS + os.path.basename(filename), 'r', encoding='latin1') as document:
        string = document.read()
    return string
###############################################################################
#
# PRIORS
#
###############################################################################
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
        if candidate not in result:
            result[candidate] = "temp"
            distribution.append(0)            
                
    for i in range(len(distribution)):
        val = distribution[i]/sum(distribution)
        result[candidates[i]] = val

    return result
###############################################################################
#
# WEIGHTS
#
###############################################################################

def compute_weights_embeddings(candidate1, candidate2):
    #when producing vectors for sentences, spacy takes the average of the vectores of the words
    c1_t = emb(candidate1)
    c2_t = emb(candidate2)

    return round((1 - c1_t.similarity(c2_t))*10, 1)
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
                if weights == EMB:
                    graph.add_edge(pair[0], pair[1], weight=compute_weights_embeddings(pair[0], pair[1]))
                break

    return graph

###############################################################################
#
# FEATURIZING
#
###############################################################################

#Returns an ordered list of lists, where each candidate has its correspondent unweighted tfidf score
def unweighted_tfidf_score(candidates, idf):
    w_tfidf = dict()
    for cand in candidates:
        w_tfidf[cand[1]] = cand[0] * idf[cand[1]]

    return sorted(w_tfidf.items(), key=lambda x: x[1], reverse=True)


def position_in_document(filename, candidates):
    pos = dict()
    with open(preprocDS + os.path.basename(filename), 'r', encoding = 'latin-1') as file:
        text = file.read() 
    
    for cand in candidates:
        if(cand[1] in text):
            pos[cand[1]] = text.index(cand[1])
        else:
            pos[cand[1]] = len(text) + 1
        
    return sorted(pos.items(), key=lambda x: x[1], reverse=False)

#Returns an ordered list of lists, where each candidate has its correspondent graph centrality score
def centrality_scores(filename):
    
    candidates, candidates_with_tf = ngrams(filename, 1, 3)
    sentences = get_sentences(filename)
    prior = compute_prior_lp(candidates, sentences)
    candidates = prior.keys()
    graph = nx.Graph()
    graph = add_graph_nodes(graph, candidates)
    graph = add_graph_edges(graph, candidates, sentences, EMB)
    pr = nx.pagerank(graph, max_iter=50, alpha=0.5, personalization=prior)

    return sorted(pr.items(), key=lambda x: x[1], reverse=True)

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

#receives a dictionary: {'candX':[rank_tfidf, rank_pr], ..}
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
scraper()
preproc_all_txt()

docs = files_to_list(preprocDS)
all_docs_idf = idf(docs, 1, 3)

final_res = dict()

for preprocessed in glob.glob(preprocDS + '*.txt'):
    candidates, candidates_with_tf = ngrams(preprocessed, 1, 3)
    
    unweighted_tfidf = unweighted_tfidf_score(candidates_with_tf, all_docs_idf)
    pagerank = centrality_scores(preprocessed)
    pos = position_in_document(preprocessed, candidates_with_tf)
    
    features = (pos, pagerank, unweighted_tfidf)
    
    ranks = candidates_as_ranks(features,candidates_with_tf)
    scores = RRFS(ranks)
    
    text_and_keyphrases = value_for_file(preprocessed, scores, 5)
    final_res[os.path.basename(preprocessed)] = text_and_keyphrases

print(final_res)


with open('data.json', 'w') as fp:
    json.dump(final_res, fp, sort_keys=True, indent=4, separators=(',', ': '))
    
print("done")