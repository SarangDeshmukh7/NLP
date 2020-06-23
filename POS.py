import nltk

paragraph = """India is a great country where people speak different languages but the national language is Hindi.
India is full of different castes, creeds, religion, and cultures but they live together. That’s the reasons 
India is famous for the common saying of “unity in diversity“. India is the seventh-largest country in the whole world."""
               
               
# POS Tagging
words = nltk.word_tokenize(paragraph)

tagged_words = nltk.pos_tag(words)

# Tagged word paragraph
word_tags = []
for tw in tagged_words:
    word_tags.append(tw[0]+"_"+tw[1])

tagged_paragraph = ' '.join(word_tags)
