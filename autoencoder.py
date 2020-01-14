# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 18:37:46 2020

@author: Kevin
"""
import keyword_model as km
import plos_preprocess as pp
import numpy as np
from scipy.spatial import distance as dist
from sklearn import metrics
import json


def findnearestn(rep, corpus_reps, n):
    distances=list()
    for i in range(len(corpus_reps)):
        distances.append(dist.cityblock(rep, corpus_reps[i]))
    distmap={}
    for index in range(len(distances)):
        distmap[str(index)]= -distances[index]
    return km.topncounts(distmap, n)

def printnearestn(rep, input_text, input_abstract, n, corpus_titles, corpus_abstracts, feature_map, corpus_reps):
    gram_size=len(feature_map[0].split())
    if n>len(corpus_reps):
        print('Error: n is greater than the total number of files in the database.')
        return -1
    matches=findnearestn(rep, corpus_reps, n)
    print('----------------------------')
    print('Input text:')
    print(input_text)
    print(input_abstract)
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
            print(corpus_abstracts[index])
        except:
            print('No abstract available.')


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers
from tensorflow.keras.layers import GaussianDropout as noise

skew=3
def bintocat(featreps):
    newreps=list()
    for rep in featreps:
        temp=list()
        for i in rep:
            if i==0:
                temp.append(1/skew)
                temp.append(2-1/skew)
            else:
                temp.append(2-1/skew)
                temp.append(1/skew)
        newreps.append(temp)
    return newreps

def invert(featreps):
    return 1-1*featreps

f_in='sigmoid'
f_act='relu'
f_out='sigmoid'
def get_compiled_model(numvar):
    model = Sequential()
    model.add(Dense(64, kernel_initializer='uniform', activation=f_in, input_shape=(numvar,)))
    model.add(Dense(16, kernel_initializer='uniform', activation=f_act, input_shape=(64,)))
    model.add(noise(.1))
    model.add(Dense(64, kernel_initializer='uniform', activation=f_act, input_shape=(16,)))
    model.add(Dense(numvar, kernel_initializer='uniform', activation=f_out, input_shape=(64,)))
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.8, nesterov=True)
    model.compile(loss='logcosh', optimizer=sgd)
    return model

##import plos bio data

#path='./quant_bio/'
#titles, rawabs, cleanabs=pp.getallabs(path,1231230)

##import arxiv ml data

#with open('C:/Startup/arxiv_abstracts/arxivData.json', 'r') as myfile:
#    rawdata=myfile.read()
#rawdata=json.loads(rawdata)
#raw=list(rawdata)[0:10000]
#titles=list()
#rawabs=list()
#cleanabs=list()
#for i in range(len(raw)):
#    titles.append(raw[i]['title'])
#    rawabs.append(pp.cleanformat(raw[i]['summary']))
#    cleanabs.append(pp.cleantext(raw[i]['summary']))

##extract ngram features

#featmap, featreps=km.corpusmodel(cleanabs, 200, 500, 2)
#print(featmap)

##train and test autoencoder
    
#i=3187 #Incorporating Domain Knowledge in Matching Problems via Harmonic Analysis
#i=4710 #Combining Evaluation Metrics via the Unanimous Improvement Ratio and its Application to Clustering Tasks
#i=7837 #Meta Reinforcement Learning with Latent Variable Gaussian Processes
i=1783 #Toward Interpretable Topic Discovery via Anchored Correlation Explanation
reps=bintocat(featreps)
data=np.asarray(reps)
model = get_compiled_model(len(reps[0]))
model.fit(data, data, epochs=20,verbose=0)
nnreps=model.predict(data)

#printnearestn(data[i], titles[i], rawabs[i], 5, titles, rawabs, featmap, data)
printnearestn(nnreps[i], titles[i], rawabs[i], 5, titles, rawabs, featmap, nnreps)
