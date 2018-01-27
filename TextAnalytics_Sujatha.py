
# coding: utf-8

# SPAM or HAM Classification Problem

# In[ ]:

#Load Libraies


# In[100]:

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

get_ipython().magic('matplotlib inline')


# In[101]:

import nltk
import string
from nltk import word_tokenize
from nltk.util import bigrams, trigrams
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter


# In[102]:

#Access File


# In[103]:

df= pd.read_csv('C:\\Users\\admin\\Desktop\\PGDM\\Text analytics\\spam_ham_dataset.csv',encoding='latin-1')
df.head()


# In[104]:

#Removing Unnamed Columns
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df = df.rename(columns={"v1":"Class", "v2":"Message"})


# In[105]:

df.head()
df.tail()


# In[106]:

#Preprocessing


# In[107]:

NLTK_STOPWORDS = set(stopwords.words('english'))

#Removing digits in text
def remove_numbers_in_string(s):
      
    return s.translate(string.digits)
#Lower case Conversion
def lowercase_remove_punctuation(s):
    print (s)
    s = s.lower()
    s = s.translate(string.punctuation)
    return s

#Remove StopWords  
def remove_stopwords(s):
    token_list = nltk.word_tokenize(s)
    exclude_stopwords = lambda token : token not in NLTK_STOPWORDS
    return ' '.join(filter(exclude_stopwords, token_list))

#Stemming

def stem_token_list(token_list):
    STEMMER = PorterStemmer()
    return [STEMMER.stem(tok) for tok in token_list]

def restring_tokens(token_list):
    return ' '.join(token_list)

def preprocessing(s):
    s = remove_numbers_in_string(s)
    s = lowercase_remove_punctuation(s)
    s = remove_stopwords(s)
    token_list = nltk.word_tokenize(s)
    token_list = stem_token_list(token_list)
    return restring_tokens(token_list)


# In[108]:

initial_features = ['Class', 'Message']
df_p = df[initial_features]
df_p['Message'] = df_p['Message'].apply(preprocessing)

for idx in range(5):
    print (df_p.Message[idx])
    print ()


# In[109]:

#Convert it to Lowercase


# In[110]:

#Remove stop words


# In[111]:

#Normalization
def normalize(s):
    for p in string.punctuation:
        s = s.replace(p, '')
 
    return s.lower().strip()


# In[112]:


#Use sentence tokenizer from NLTK
from nltk.tokenize import sent_tokenize
for idx in range(5572):
    text=df_p.Message[idx]
    text=normalize(text)
    sent_tokenize_list = sent_tokenize(text)
    
    len(sent_tokenize_list)
    #print(sent_tokenize_list)
    #print ()


# In[113]:

#Load sklearn Libraries
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import *
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, roc_auc_score


# In[114]:

#Modelling
TEST_SIZE = 0.40
train_X, test_X, train_y, test_y = train_test_split(df_p.Message,df_p.Class,test_size=TEST_SIZE,random_state=42)
    


# In[115]:

#Convert: Converting Raw text into numerical fetures (BOW, TF-IDF and W2V)
#http://datameetsmedia.com/bag-of-words-tf-idf-explained/
bag_of_words_vectorizer = CountVectorizer(analyzer = "word",
                                          tokenizer = None,    
                                          preprocessor = None,
                                          ngram_range = (1, 1),
                                          binary = False,
                                          strip_accents='unicode')


# In[116]:

bow_feature_matrix_train = bag_of_words_vectorizer.fit_transform(train_X)
bow_feature_matrix_test = bag_of_words_vectorizer.transform(test_X)
bow_feature_matrix_train, bow_feature_matrix_test


# In[117]:

#Multinominla NaiveBayes
multinomial_nb_classifier = MultinomialNB()
multinomial_nb_classifier.fit(bow_feature_matrix_train, train_y)
multinomial_nb_prediction = multinomial_nb_classifier.predict(bow_feature_matrix_test)


# In[118]:

#Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report


# In[119]:

print(confusion_matrix(test_y,multinomial_nb_prediction))
print('\n')
print(classification_report(test_y,multinomial_nb_prediction))


# In[120]:

##Tf-IDF
from sklearn.feature_extraction.text import  TfidfTransformer


# In[121]:

from sklearn.pipeline import Pipeline


# In[122]:

pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[123]:

X=df_p.Message
y=df_p.Class
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4,random_state=42)


# In[124]:

# May take some time
pipeline.fit(X_train,y_train)


# In[125]:

predictions = pipeline.predict(X_test)


# In[126]:

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[127]:

#########  Lexicon Model


# In[128]:

lex_file = open("C:\\Users\\admin\\Desktop\\PGDM\\Text analytics\\AFINN-111.csv")


# In[129]:

lexicons = {}
records = lex_file.readlines()
for record in records:
    #print(record) - line contains newline charecter
    #print(record.rstrip('\n').split(",")) - to remove new line charecter
    lexicons[record.rstrip('\n').split(",")[0]] = int(record.rstrip('\n').split(",")[1])
#print(lexicons)
#lexicons["abandon"]


# In[135]:

#Strip the records and create a word list for each tweet
word_list = []
for idx in range(5572):
    text=df_p.Message[idx]
    text=normalize(text)
    words = []
    token_list = nltk.word_tokenize(text)
    #print(tokens)
    for token in token_list:
        words.append(token.lower())
    word_list.append(words)  
    


# In[136]:

word_list


# In[137]:

strength = []
for tweet in word_list:
    score = 0
    for word in tweet:
        if word in (lexicons):
            score = score + lexicons[word]
    strength.append(score)


# In[138]:

strength


# In[139]:

for idx in range(5572):
   
    
    print(df_p.Message[idx], df_p.Class[idx], strength[idx])


# In[140]:

strength=np.array(strength)


# In[141]:

strength


# In[142]:

df_strength=pd.DataFrame(strength)


# In[150]:

df_p['Strength']=df_strength[0]


# In[151]:

df_p


# In[169]:

df_p['Predicted_Class']=df_p['Strength'].apply(lambda x:"ham" if x>=0 else "spam")


# In[170]:

df_p


# In[171]:

total=0
for idx in range(5572):
    
    if df_p.Class[idx]==df_p.Predicted_Class[idx]:
        total=total+1
        #print(total)
        
accuracy=total/5572
accuracy
        


# In[172]:

total


# In[ ]:



