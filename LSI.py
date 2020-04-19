# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 02:56:23 2020

@author: Kevin
"""
import keyword_model as km
from sklearn.decomposition import PCA

class Document:
    def __init__(self, title, text, vector):
        self.title=title
        self.text=text
        self.vector=vector

class LSI:
    
    def __init__(self, titles, raw_texts, n_features, gram_size):
        clean_texts=list()
        for entry in raw_texts:
            clean_texts.append(km.cleantext(entry))
        feature_map=km.makefeatmap(clean_texts, n_features, gram_size)
        feature_reps=[km.extractfeat(text, feature_map) for text in clean_texts]
        pca=PCA()
        pca_reps=pca.fit_transform(feature_reps)
        doclist=list()
        for i in range(len(raw_texts)):
            doclist.append(Document(titles[i], raw_texts[i], feature_reps[i]))
        self.pca=pca
        self.feature_map=feature_map
        self.database=doclist
        self.lsi_reps=pca_reps
        
    #Constructs a vector representation using the model
    def transform(self, text):
        return self.pca.transform([km.extractfeat(text, self.feature_map)])
    
    def getmatches(self, input_text, corpus_model, n):
        clean=km.cleantext(input_text)
        inputfeat=self.transform(clean)
        return km.alldist(inputfeat, self.doc_vectors, n)
        
    #takes input text and prints the n most relevant articles
    def printmatches(self, input_text, corpus_model, n):
        corpus_reps=self.feature_map
        corpus_texts=corpus_model[2]
        corpus_titles=corpus_model[3]
        if n>len(corpus_reps):
            print('Error: n is greater than the total number of files in the database.')
            return -1
        matches=km.getmatches(input_text, corpus_model, n)
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
                
    #Returns shared keywords between two texts
    def intersection(self, text1, text2, corpus_model):
        common_grams=list()
        text1=km.cleantext(text1)
        text2=km.cleantext(text2)
        feat1=km.getgrams(self.feature_map, km.extractfeat(text1, self.feature_map))
        feat2=km.getgrams(self.feature_map, km.extractfeat(text2, self.feature_map))
        for word in feat1:
            if word in feat2:
                common_grams.append(word)
        return common_grams
        
def main():
    titles, abstracts=km.load_arxiv(100)
    test=LSI(titles, abstracts, 10, 1)
    print(test.lsi_reps[0])

if __name__=="__main__":
    main()
