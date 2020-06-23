import nltk
import re
import heapq
import numpy as np

paragraph = """India is a great country where people speak different languages but the national language is Hindi.
India is full of different castes, creeds, religion, and cultures but they live together. That’s the reasons 
India is famous for the common saying of “unity in diversity“. India is the seventh-largest country in the whole world."""
               
               
# Tokenize sentences
dataset = nltk.sent_tokenize(paragraph)
for i in range(len(dataset)):
    dataset[i] = dataset[i].lower()
    dataset[i] = re.sub(r'\W',' ',dataset[i])
    dataset[i] = re.sub(r'\s+',' ',dataset[i])


# Creating word histogram
word2count = {}
for data in dataset:
    words = nltk.word_tokenize(data)
    for word in words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1
            
# Selecting best 100 features
freq_words = heapq.nlargest(100,word2count,key=word2count.get)


# IDF Dictionary
word_idfs = {}
for word in freq_words:
    doc_count = 0
    for data in dataset:
        if word in nltk.word_tokenize(data):
            doc_count += 1
    word_idfs[word] = np.log(len(dataset)/(1+doc_count))
    
# TF Matrix
tf_matrix = {}
for word in freq_words:
    doc_tf = []
    for data in dataset:
        frequency = 0
        for w in nltk.word_tokenize(data):
            if word == w:
                frequency += 1
        tf_word = frequency/len(nltk.word_tokenize(data))
        doc_tf.append(tf_word)
    tf_matrix[word] = doc_tf
    
# Creating the Tf-Idf Model
tfidf_matrix = []
for word in tf_matrix.keys():
    tfidf = []
    for value in tf_matrix[word]:
        score = value * word_idfs[word]
        tfidf.append(score)
    tfidf_matrix.append(tfidf)   
    
# Finishing the Tf-Tdf model
X = np.asarray(tfidf_matrix)

X = np.transpose(X)
