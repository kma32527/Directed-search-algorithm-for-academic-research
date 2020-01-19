# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 14:41:58 2019

@author: Kevin Ma
"""
from scipy import spatial
import random
import json
import math
import re
import nltk
from nltk.tokenize import word_tokenize as wt
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
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


with open('C:/Startup/arxiv_abstracts/arxivData.json', 'r') as myfile:
    rawdata=myfile.read()
numdoc=len(rawdata)
rawdata=json.loads(rawdata)
data=list(rawdata[0:numdoc])

#Removes special characters and forces lower case
def cleanText(paper):
    paper=re.sub('\[[^\]\[]*\]', '', str(paper))
    paper=re.sub('<sub>', '_', paper)
    paper=re.sub('</sub>', '', paper)
    paper=re.sub('\s{1,}', ' ', paper)
    paper=re.sub('-', '', paper)
    paper=re.sub('<title>.*</title>', '', paper)
    paper=re.sub('<[^<>]*>', ' ', paper)
    paper=re.sub('[(~%)\-/,=:;+``#:.]', '', paper)
    paper=re.sub('b\'', '', paper)
    paper=re.sub('n\'', '', paper)
    paper=re.sub('\\\\', '', paper)
    paper=re.sub('\'', '', paper)
    return paper.lower()

#Cleans text of each data entry
#Destructive
def cleanData(data):
    for i in range(len(data)):
        data[i]['summary']=re.sub('\s{1,}', ' ', re.sub('\\\n|\'|["/?%{}&`$\-()_:;,+.]|\\\\|\]|\[', ' ', data[i]['summary']).lower())
    return data

#Extracts word counts from across all data
#Option to restrict to parts of speech most associated with key words
#Coded to be nouns, adjectives, adverbs, verbs
#Returns n most frequent word, where n is an integer
def corpusCounts(data, n, pos):
    corpuscat=''
    goodpos=['N', 'J', 'R', 'V']
    if pos==False:
        goodpos='QWERTYUIOPASDFGHJKLZXCVBNM'
    for i in range(len(data)):
        corpuscat+=data[i]['summary']
    tokens=nltk.pos_tag(wt(corpuscat))
    fdist1 = FreqDist([token[0] for token in tokens if len(token[0])>1 and token[1][0] in goodpos and not token[0] in stopwords])
    features=fdist1.most_common(n)
    return features

def keywithmaxval(d):
    """ a) create a list of the dict's keys and values;
        b) return the key with the max value"""
    v=list(d.values())
    k=list(d.keys())
    return k[v.index(max(v))]

#Returns the top n words ranked by their associated word counts
def getNMax(counts, n):
    maxlist=list()
    for i in range(min(n, len(counts))):
        if len(counts) > 0:
            max_key=keywithmaxval(counts)
            maxlist.append([max_key, counts[max_key]])
            counts.pop(max_key)
    return maxlist

#Concatenates all of the summaries into a single text
def concatText(textlist):
    cattext='';
    for text in textlist:
        cattext+=text['summary']
    return cattext

#Extracts ngrams from text
def getNgramCounts(text, numgram):
    freqs={}
    split=text.split()
    gramlist=list()
    for i in range(2, numgram+1):
        gramlist.append(nltk.ngrams(split, i))
    for ngram in gramlist:
        for gram in ngram:
            if not any(token in stopwords or len(token) < 2 for token in gram):
                gram=" ".join(gram)
                if gram in freqs:
                    freqs[gram]+=1
                else:
                    freqs[gram]=1
    return freqs

def makeEmbeddingMap(text, corpusCounts, numgram, maxperkey):
    if numgram<2:
        return -1
    freqs=getNgramCounts(text, numgram)
    topgrams=getNMax(freqs, 2000)
    stemmedWords={}
    onlyStem={}
    for entry in topgrams:
        tokens=wt(entry[0])
        tokens=[stemmer.stem(token) for token in tokens]
        stem=" ".join(tokens)
        if stem in stemmedWords:
            stemmedWords[stem][entry[0]]=entry[1]
            onlyStem[stem]+=entry[1]
        else:
            stemmedWords[stem]={}
            stemmedWords[stem][entry[0]]=entry[1]
            onlyStem[stem]=entry[1]
    topgrams=getNMax(onlyStem, 1500)
    vectorMap={}
    for word in corpusCounts:
        vectorMap[word[0]]=list()
        for gram in topgrams:
            if len(vectorMap[word[0]])-1 < maxperkey and word[0] in gram[0] and not gram[0] in vectorMap[word[0]]:
                vectorMap[word[0]].append(gram[0])
    return vectorMap

##Constructs 3 different types of vector representations for comparison
##Faster to do all 3 at once when doing a large batch
def makeFeatureVector(text, vectorMap):
    textgrams=getNgramCounts(text, 3)
    textstems={}
    for entry in textgrams:
        tokens=wt(entry)
        tokens=[stemmer.stem(token) for token in tokens]
        stem=" ".join(tokens)
        if stem in textstems:
            textstems[stem] += 1
        else:
            textstems[stem]=1
    ##binvector is a binary vector with a value in {0, 1} for each key
    ##countvector is a count of each ngram
    ##maxvector keeps only the maximum occurring ngram for each stem
    binvector=list()
    countvector=list()
    maxvector=list()
    for key in vectorMap:
        binrep=[0]*len(vectorMap[key])
        countrep=[0]*len(vectorMap[key])
        maxrep=[0]*(len(vectorMap[key]))
        maxstemcount=-1
        maxstemind=-1
        for i in range(len(vectorMap[key])):
            trialstem=vectorMap[key][i]
            if trialstem in textstems:
                stemcount=textstems[trialstem]
                binrep[i]=1
                countrep[i]=stemcount
                if stemcount>maxstemcount:
                    maxstemcount=stemcount
                    maxstemind=i
        if maxstemind > -1:
            maxrep[maxstemind]=1
        binvector+=binrep
        countvector+=countrep
        maxvector+=maxrep
    return binvector, countvector, maxvector

def getKeyMap(vectorMap):
    keymap=list()
    for key in vectorMap:
        for stem in vectorMap[key]:
            keymap.append(stem)
    return keymap


#data=cleanData(data)
#features=corpusCounts(data, 500, pos=True)
#vectorMap=makeEmbeddingMap(concatText(data), features, 3, 20000)
#binfeaturereps=list()
#countfeaturereps=list()
#maxfeaturereps=list()
#for i in range(len(data)):
#    binreps, countreps, maxreps=makeFeatureVector(data[i]['summary'], vectorMap)
#    binfeaturereps.append(binreps)
#    countfeaturereps.append(countreps)
#    maxfeaturereps.append(maxreps)
#normfeaturereps=list()
#for i in range(len(countfeaturereps)):
#    entry=countfeaturereps[i]
#    length=len(wt(data[i]['summary']))
#    entry=[-math.log((value+1)/(length+2)) for value in entry]
#    normfeaturereps.append(entry)
##------------------------------------------------
testdoc=random.randint(1,len(data))
featurereps=list(maxfeaturereps)
#featurereps2=list(maxfeaturereps)

##-------------------------------------------------
#from sklearn.decomposition import PCA
#pca=PCA(n_components=100)
#pca.fit_transform(featurereps)
#print(len(featurereps[0]))
##----------------------------------------------------
#testdoc=25306
#testdoc=1147
#testdoc=26980
#testdoc=33217
#testdoc=15292
#inputtext='This has a very simple combinatorial underlying structure, and ultimately one can regard a k-simplex as an arbitrary set of k+1 objects with faces (and faces of faces etc.) given by appropriately sized subsets – one can always provide a “geometric realization” of this abstract set description by constructing the corresponding geometric simplex.'
#testtext=cleanText(inputtext)
#bins, counts, maxes=makeFeatureVector(testtext, vectorMap)
test=featurereps[testdoc]
#test=maxes
all_differences=list()
#print(featurereps[testdoc])
for i in range(len(featurereps)): #
    all_differences.append(spatial.distance.sokalmichener(test, featurereps[i]))
distmap={}
for i in range(len(all_differences)):
    distmap[str(i)]=-all_differences[i]
#
matchArray=getNMax(distmap, 10)
print('----------------------------')
print('Input abstract: ' + rawdata[testdoc]['title'])
#print('Input Abstract: Learning Multi-Task Communication with Message Passing for Sequence Learning')
print('------')
#print(inputtext)
print(rawdata[testdoc]['summary'])
for i in range(len(matchArray)):
    print('----------------------------')
    print('Rank: ' + str(i+1) + ' Index: ' + str(matchArray[i][0]) + ' Distance=' + str(all_differences[int(matchArray[i][0])]))
    print('Title: ' + rawdata[int(matchArray[i][0])]['title'])
    print('------')
    print(rawdata[int(matchArray[i][0])]['summary'])
##-----------------------------------------------------
#print(len(featurereps[0]))
#keymap=getKeyMap(vectorMap)
#print(keymap)
#all_differences=list()
#for i in range(len(featurereps)):
#    all_differences.append(spatial.distance.cosine(list(clust.cluster_centers_[0]), featurereps[i]))
#distmap={}
#for i in range(len(all_differences)):
#    distmap[str(i)]=-all_differences[i]
#
#matchArray=getNMax(distmap, 5)
#print('----------------------------')
#print('Input abstract: ')
#print(rawdata[int(matchArray[0][0])]['title'])
#print('------')
#print(rawdata[testdoc]['summary'])
#for i in range(len(matchArray)):
#    print('----------------------------')
#    print('Rank: ' + str(i) + ' Index: ' + str(matchArray[i][0]) + ' Distance=' + str(all_differences[int(matchArray[i][0])]))
#    print(rawdata[int(matchArray[i][0])]['title'])
#    print('------')
#    print(rawdata[int(matchArray[i][0])]['summary'])
