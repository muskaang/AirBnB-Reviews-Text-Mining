
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
import re
import heapq

df1 = pd.read_csv('cleaned1.csv', names = ['listing_id','review_date','review'])
df2 = pd.read_csv('cleaned2.csv', names = ['listing_id','review_date','review'])
df3 = pd.read_csv('cleaned3.csv', names = ['listing_id','review_date','review'])
df4 = pd.read_csv('cleaned4.csv', names = ['listing_id','review_date','review'])
# print(df.head(10))
df = pd.concat([df1,df2,df3,df4])
print(df.shape[0])
by_listing = df.groupby(['listing_id'])
stopwords = nltk.corpus.stopwords.words('english')

all_summaries = {}
for x in by_listing.groups:
    listing_data = by_listing.get_group(x)
    reviews = listing_data['review']
    listing_id_iter = listing_data['listing_id']
    listing_id = listing_id_iter.values[0]
    combined_reviews = ''
    for review in reviews.iteritems():
        combined_reviews += str(review[1])
        
    sentences = re.split('[,.?]', combined_reviews)
#     for sentence in sentences:
#         print(sentence)

    word_frequencies = {}  
    for word in nltk.word_tokenize(combined_reviews):  
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
                
    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)
    
    sentence_scores = {}  
    for sent in sentences:
        sent = sent.strip()
        for word in nltk.word_tokenize(sent):
            if word in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]

    fout = './sentences.txt'
    fo = open(fout, 'a')
    for k in sentence_scores.keys():
        fo.write(k + '\n\n')
    fo.close()
     
    summary_sentences = heapq.nlargest(3, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    print(summary, end = '\n\n')
    all_summaries[listing_id] = summary

fout2 = './summaries.csv'
fo2 = open(fout2, 'w')
for k,v in all_summaries.items():
    fo2.write(str(k) + ',' + v + '\n')
fo2.close()

