##Algorithm functions

import re
import json
import math
from os import listdir
from os.path import isfile, join
import nltk
from nltk.tokenize import word_tokenize as wt
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy import spatial
#import plos_preprocess as plos

#PorterStemmer used for lemmatization of words 
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

#Loads arXiv abstracts
def load_arxiv(*args):
    with open('C:/Startup/arxiv_abstracts/arxivData.json', 'r') as myfile:
        arxiv_data=myfile.read()
    arxiv_data=json.loads(arxiv_data)
    if len(args)>0 and isinstance(args[0], int):
        arxiv_data=arxiv_data[:args[0]]
    abstracts=list()
    titles=list()
    for entry in arxiv_data:
        titles.append(entry['title'])
        abstracts.append(entry['summary'])
    return titles, abstracts

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
def kmtokens(text, *args):
    if len(args)==0:
        tokens=wt(text)
        return [stemmer.stem(token.strip()) for token in tokens if len(token)>1 and not token in stopwords]
    if args[0]=='pos':
        goodpos=['N', 'J', 'R', 'V']
        tokens=nltk.pos_tag(wt(text))
        return [stemmer.stem(token[0].strip()) for token in tokens if len(token[0])>1 and not token[0] in stopwords]

#returns the n most frequent words across the corpus
def corpuswc(data, n, *args):
    megatext=concat(data)
    freqs = FreqDist(kmtokens(megatext))
    features=freqs.most_common(n)
    return features

def gettfidf(data, tflist, n):
    tfidfmap={}
    dfmap=docfreq(data, tflist)
    for entry in tflist:
        word=entry[0]
        tf=entry[1]
        df=dfmap[word]
        tfidfmap[word]=math.log(1/df)*tf
    return topncounts(tfidfmap,len(tfidfmap))

def docfreq(data, word_counts):
    dfcounts={}
    datatokens=list()
    for text in data:
        datatokens.append(kmtokens(text))
    for entry in word_counts:
        word=entry[0]
        count=0
        for i in range(len(datatokens)):
            if len(datatokens[i])>0:
                numwords=len(datatokens[i])
                datatokens[i]=[token for token in datatokens[i] if not token==word]
                if len(datatokens[i])<numwords:
                    count+=1
        dfcounts[word]=count
    return dfcounts

#Returns the max key by value
def maxkey(d):
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

#Returns a list of ngrams by count of a given size 
def ngramcounts(tokens, gram_size):
    freqs={}
    ngrams=nltk.ngrams(tokens,gram_size)
    for ngram in ngrams:
        gram=" ".join(ngram)
        if gram in freqs:
            freqs[gram]+=1
        else:
            freqs[gram]=1
    return freqs

#Creates a map consisting of word stems and a list of valid ngrams associated with the stem
def makefeatmap(text_list, n_features, gram_size):
    word_counts=corpuswc(text_list, n_features)
    top_words=[entry[0] for entry in word_counts]
    text=concat(text_list)
    tokens=kmtokens(text)
    tokens=[word for word in tokens if word in top_words]
    if gram_size==1:
        allfreqs={}
        for entry in word_counts:
            allfreqs[entry[0]]=entry[1]
    else:
        allfreqs=ngramcounts(tokens, gram_size)
    nmax=topncounts(allfreqs, n_features)
    feature_map=[entry[0] for entry in nmax]
    return feature_map

#Constructs a vector representation of input text using the categorical embedding map
def extractfeat(text, feature_map):
    if len(text)==0:
        return -1
    tokens=kmtokens(cleantext(text))
    gram_size=1
    if gram_size > 1:
        textgrams=ngramcounts(tokens, gram_size)
    else:
        textgrams=tokens    
    vector=list()
    for gram in feature_map:
        if gram in textgrams:
            vector.append(1)
        else:
            vector.append(0)
    return vector

#Returns the indeces of the n nearest most relevant articles found in the corpus
def alldist(rep, corpus_reps, n):
    dist={}
    for i in range(len(corpus_reps)):
        dist[str(i)]=-spatial.distance.cosine(rep, corpus_reps[i])
    alldists=topncounts(dist, n)
    for i in range(len(alldists)):
        alldists[i][1]=-alldists[i][1]
    return alldists

def getmatches(input_text, corpus_model, n):
    clean=cleantext(input_text)
    featmap=corpus_model[0]
    inputfeat=extractfeat(clean, featmap)
    return alldist(inputfeat, corpus_model[1], n)

#takes input text and prints the n most relevant articles
def printmatches(input_text, corpus_model, n):
    feature_map=corpus_model[0]
    corpus_reps=corpus_model[1]
    corpus_texts=corpus_model[2]
    corpus_titles=corpus_model[3]
    gram_size=len(feature_map[0].split())
    if n>len(corpus_reps):
        print('Error: n is greater than the total number of files in the database.')
        return -1
    matches=getmatches(input_text, corpus_model, n)
    print('----------------------------')
    print('Input text:')
    print(input_text)
    print('----------------------------')
    for i in range(len(matches)):
        index=int(matches[i][0])
        distance=matches[i][1]
        print('----------------------------')
        print('Rank: ' + str(i+1) + ' Index: ' + str(index) + ' Distance=' + str(distance))
        print('Title: ' + corpus_titles[index])
        print('------')
        try:
            print('Text:')
            print(corpus_texts[index])
        except:
            print('No abstract available.')

#Creates a model from input parameters
#feature_map is an array of ngrams
#feature_reps consists of the bag of words representations of all articles in the corpus  
def corpusmodel(titles, raw_texts, n_features, gram_size):
    clean_texts=list()
    for entry in raw_texts:
        clean_texts.append(cleantext(entry))
    feature_map=makefeatmap(clean_texts, n_features, gram_size)
    feature_reps=list()
    for i in range(len(clean_texts)):
        feature_reps.append(extractfeat(clean_texts[i], feature_map))
    return [feature_map, feature_reps, raw_texts, titles]

#Returns all valid ngrams associated with a word
def peekgrams(feature_map, word):
    stem=stemmer.stem(word.lower().strip())
    if stem in feature_map:
        return feature_map[stem]

def intersection(text1, text2, corpus_model):
    feature_map=corpus_model[0]
    common_grams=list()
    text1=cleantext(text1)
    text2=cleantext(text2)
    feat1=getgrams(feature_map, extractfeat(text1, feature_map))
    feat2=getgrams(feature_map, extractfeat(text2, feature_map))
    for word in feat1:
        if word in feat2:
            common_grams.append(word)
    return common_grams

#Returns list of ngrams associated to non-zero elements of a feature vector
def getgrams(feature_map, vector):
    hasgrams=list()
    for i in range(len(vector)):
        if vector[i]>0:
            hasgrams.append(feature_map[i])
    return hasgrams
            
#titles, abstracts=load_arxiv(100)
#test=corpusmodel(abstracts, 10, 1, titles)
#printmatches(abstracts[0], test, 5)
# path='./quant_bio/'
# titles, rawabs, cleanabs=plos.getallabs(path,300)
# featmap, featreps=corpusmodel(cleanabs, 100, 500, 2)
# test=0
# print(titles[test])
# printnearestn(rawabs[test],10, titles, rawabs,featmap,featreps)
