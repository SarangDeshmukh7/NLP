import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

paragraph = """India is a great country where people speak different languages but the national language is Hindi.
India is full of different castes, creeds, religion, and cultures but they live together. That’s the reasons 
India is famous for the common saying of “unity in diversity“. India is the seventh-largest country in the whole world."""
               
               
sentences = nltk.sent_tokenize(paragraph)


# Removing stopwords
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [word for word in words if word not in stopwords.words('english')]
    sentences[i] = ' '.join(words) 
