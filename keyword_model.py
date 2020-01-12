##Algorithm functions

import re
import math
from os import listdir
from os.path import isfile, join
import nltk
from nltk.tokenize import word_tokenize as wt
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy import spatial
import plos_preprocess as plos

stemmer = PorterStemmer()
stopwords=set(stopwords.words('english'))
stopwords=[word for word in stopwords]
stopwords.append('et')
stopwords.append('al')
stopwords.append('github')
stopwords.append('https')
stopwords.append('www')
stopwords.append('com')
stopwords.append('dnn')
stopwords.append('cnn')
stopwords.append('survey')
stopwords.append('review')

# Strips special characters from text and converts to lower case
def cleantext(paper):
    paper=re.sub('\[[^\]\[]*\]', '', str(paper))
    paper=re.sub('<sub>', '_', paper)
    paper=re.sub('</sub>', '', paper)
    paper=re.sub('\s{1,}|-', ' ', paper)
    paper=re.sub('<title>.*</title>', '', paper)
    paper=re.sub('<[^<>]*>', ' ', paper)
    paper=re.sub('[(~%)\-/,=:;+``#:.]', '', paper)
    paper=re.sub('b\'', '', paper)
    paper=re.sub('n\'', '', paper)
    paper=re.sub('\\\\', '', paper)
    paper=re.sub('\'', '', paper)
    return paper.lower().strip()

#Generates list of terms frequency of nouns, adjectives, adverbs, and verbs from the set of all papers
#Returns the n most frequent terms
def postokens(text):
    goodpos=['N', 'J', 'R', 'V']
    tokens=nltk.pos_tag(wt(text))
    return [stemmer.stem(token[0].strip()) for token in tokens if len(token[0])>1 and token[1][0] in goodpos and not token[0] in stopwords]

def corpuswc(data, n):
    text=concat(data)
    freqs = FreqDist(postokens(text))
    features=freqs.most_common(n)
    return features

#Returns the max key by value
def maxkey(d):
    """ a) create a list of the dict's keys and values;
        b) return the key with the max value"""
    v=list(d.values())
    k=list(d.keys())
    return k[v.index(max(v))]

#Returns the top n keys by frequency
def topncounts(counts, n):
    max_list=list()
    for i in range(min(n, len(counts))):
        if len(counts) > 0:
            max_key=maxkey(counts)
            max_list.append([max_key, counts[max_key]])
            counts.pop(max_key)
    return max_list

#Concatenates the text together
def concat(text_list):
    cat_text='';
    for text in text_list:
        cat_text+=text
    return cat_text

#receives text input and a maximal size for n-grams
#numgram is the maximal size of n for n-grams
#outputs frequencies across ngrams
def ngramcounts(tokens, n):
    freqs={}
    ngrams=nltk.ngrams(tokens,n)
    for ngram in ngrams:
        gram=" ".join(ngram)
        if gram in freqs:
            freqs[gram]+=1
        else:
            freqs[gram]=1
    return freqs

#Creates a map consisting of word stems and a list of possible ngrams associated with the stem
#Returned map is of the form [stem]:([ngram including stem]:[count])
def makefeatmap(text_list, word_counts, gram_size, n_features):
    top_words=[entry[0] for entry in word_counts]
    if gram_size<2:
        return -1
    text=concat(text_list)
    tokens=postokens(text)
    tokens=[word for word in tokens if word in top_words]
    allfreqs=ngramcounts(tokens, gram_size)
    nmax=topncounts(allfreqs, n_features)
    db_map=[entry[0] for entry in nmax]
    return db_map

#Constructs a vector representation of input text using the categorical embedding map
def extractfeat(text, feature_map, gram_size):
    if len(text)==0:
        return -1
    vector=list()
    tokens=postokens(text)
    textgrams=ngramcounts(tokens, gram_size)
    for gram in feature_map:
        if gram in textgrams:
            vector.append(1)
        else:
            vector.append(0)
    return vector


def findnearestn(rep, corpus_reps, n):
    distances=list()
    for i in range(len(corpus_reps)):
        distances.append(spatial.distance.hamming(rep, corpus_reps[i]))
    distmap={}
    for index in range(len(distances)):
        distmap[str(index)]= -distances[index]
    return topncounts(distmap, n)

def printnearestn(input_text, n, corpus_titles, corpus_abstracts, feature_map, corpus_reps):
    gram_size=len(feature_map[0].split())
    if n>len(corpus_reps):
        print('Error: n is greater than the total number of files in the database.')
        return -1
    rep=extractfeat(cleantext(input_text), feature_map, gram_size)
    matches=findnearestn(rep, corpus_reps, n)
    print('----------------------------')
    print('Input text:')
    print(input_text)
    print('----------------------------')
    for i in range(len(matches)):
        index=int(matches[i][0])
        distance=-matches[i][1]
        print('----------------------------')
        print('Rank: ' + str(i+1) + ' Index: ' + str(index) + ' Distance=' + str(distance))
        print('Title: ' + corpus_titles[index])
        print('------')
        try:
            print('Abstract:')
#            print(dbabstracts[index])
        except:
            print('No abstract available.')

def corpusmodel(clean_texts, n_words, n_features, gram_size):
    word_counts=corpuswc(clean_texts, n_words)
    feature_map=makefeatmap(clean_texts, word_counts, gram_size, n_features)
    feature_reps=list()
    for i in range(len(clean_texts)):
        feature_reps.append(extractfeat(clean_texts[i], feature_map, gram_size))
    return feature_map, feature_reps

def peekgrams(feature_map, word):
    stem=stemmer.stem(word.lower().strip())
    if stem in feature_map:
        return feature_map[stem]

def printcontents(feature_map, vector):
    for i in range(len(vector)):
        if vector[i]>0:
            print(feature_map[i])

# path='./quant_bio/'
# titles, rawabs, cleanabs=plos.getallabs(path,300)
# featmap, featreps=corpusmodel(cleanabs, 100, 500, 2)
# test=0
# print(titles[test])
# printnearestn(rawabs[test],10, titles, rawabs,featmap,featreps)