#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Take input of all files
#Preprocess data/ files using BeautifulSoup and regular expression
# Save each email in seprate folder
# Loader File function to get all the output Files 
# 1) Implement  TF-IDF Vectoriztion
# 2) Implement Bag of words representation
# Topic Modelling LDA using Bag of Words
# 3) Implement Using uni-grams and bi-grams
# Topic Modelling LDA with uni-grams and bi-grams
# 4) Implement Word to Vector representation
# Topic Modelling with Word to Vector



# In[ ]:
import os
import sys
import subprocess
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
import gensim
from tqdm import tqdm
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sklearn
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud,STOPWORDS









# In[ ]:


path = r'C:\Users\parth\OneDrive\Desktop\NLP_EMAIL\html'

os.chdir(path)

def read_text_file(file_path):
    with open (file_path, 'r', errors='ignore') as f:
        html = f.read()
        print(f.read())
        
        fo = open("only_html.txt", "a")
        fo.write(html_txt)
        fo.close()
        
    
            
for file in os.listdir():
    if file.endswith(".html"):
        file_path = f"{path}\{file}"
        html_text =read_text_file(file_path)
        


# In[ ]:


lemmatizer = WordNetLemmatizer()
for html_file in tqdm(html): 
    f = open(html_file, 'r', encoding='utf-16-le', errors='ignore')
    html_text = f.read()
    #Preprocessing Start
    stripped_text = re.sub('<[^>]*>', '', html_text)
    stripped_text= re.sub(r"http\S+", "",html_text )
    soup = BeautifulSoup(stripped_text, 'lxml')
    text = soup.get_text()
    text = re.sub("\S*\d\S*", "", text).strip()
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    text = re.sub('&nbsp;', ' ', text)
    text = re.sub('&lt;', '', text)
    text = re.sub('&gt;', '', text)
    text = re.sub('&#128227;', '', text)
    s = text.encode('ascii',errors='ignore').decode()
    ss = s.split('\n')
    #Preprocessing end
    tk = [word_tokenize(word) for word in ss] #Tokenization
    stop_words = set(stopwords.words('english'))
    stop = [w for w in tk[0] if not w.lower() in stop_words] #Removal of stopwords
    sss =[lemmatizer.lemmatize(word) for word in stop] #lemmatization
    final_text = ' '.join(sss)
    #Writing to files
    with open(os.path.join(r'C:\Users\parth\OneDrive\Desktop\NLP_EMAIL\NP\OUTPUT_NP',html_file.split('\\')[-1][:-5]+'.txt'),'w') as t:
        t.writelines(final_text)
        t.close()


# In[ ]:


def AllFilesLoader(container_path):
    return sklearn.datasets.load_files(container_path)  

dataloader = AllFilesLoader(r'C:\Users\parth\OneDrive\Desktop\NLP_EMAIL\NP')
dataloader=[str(x) for x in actual.data if x!=b'' ]
dataloader=[str(i.replace("b'","")) for i in actual ]


# In[ ]:


dx = pd.DataFrame(actual, columns = ['text'])
txt=dx['text']


# In[ ]:


#>>>>> Using TF-IDF Vectoriztion
tfidf=TfidfVectorizer(ngram_range=(1,2), min_df=25)
tfidf_transform=tfidf.fit_transform(txt)


# In[ ]:


#>>>>> Principal Component Analysis
pca = PCA(0.85).fit_transform(tfidf_transform.toarray())



# In[ ]:


#>>>>> Kmeans Clustering
kmeans = KMeans(n_clusters=7).fit(pca)
trans=kmeans.transform(pca)
label = kmeans.predict(pca)


# In[ ]:


u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(pca[label == i , 0] , pca[label == i , 1] , label = i)
plt.legend()
plt.show()
plt.savefig("Tf_idf.png")


# In[ ]:


clusters = ['Cluster '+str(i) for i in range(7)]
docs = ['Doc '+str(i) for  i in range(txt.shape[0])]
tfidfdf = pd.DataFrame(np.round(trans,2) ,columns = clusters, index = docs)
tfidfdf['Cluster'] = np.argmax(tfidfdf.values,axis=1)
tfidfdf['Cluster']='Cluster ' + tfidfdf['Cluster'].astype(str)
tfidfdf


# In[ ]:


tfidf_clusters=tfidfdf['Cluster'].values.tolist()


# In[ ]:


tfidf = Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2), min_df=20)),('lda',LatentDirichletAllocation(n_components=7))])
tfidf_fit = tfidf.fit_transform(txt)


# In[ ]:


topics1 = ['Topic '+str(i) for i in range(tfidf['lda'].n_components)]
docs1 = ['Doc '+str(i) for  i in range(txt.shape[0])]
tfidfdf = pd.DataFrame(np.round(tfidf_fit,2) ,columns = topics1, index = docs1)
tfidfdf['Dominant topic'] = np.argmax(tfidfdf.values,axis=1)
tfidfdf['Dominant topic']='Topic ' + tfidfdf['Dominant topic'].astype(str)
tfidfdf


# In[ ]:


tfidfdf['Cluster']=tfidf_clusters
tfidfdf.to_csv('tf_idf.csv')
# In[ ]:


# >>>>>Bag of words representation
vectorizer = CountVectorizer()
vzx = vectorizer.fit_transform(txt)
vzx


# In[ ]:


# >>>>>Principal Component Analysis 
pca = PCA(0.85).fit_transform(vz.toarray())


# In[ ]:


# >>>>>Clustering using Kmeans
# Here i am selecting 7 cluster.
kmeans = KMeans(n_clusters=7).fit(pca)
trans=kmeans.transform(pca)
label = kmeans.predict(pca)


# In[ ]:


u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(pca[label == i , 0] , pca[label == i , 1] , label = i)
plt.legend()
plt.show()
plt.savefig("Bag_of_words.png")


# In[ ]:


clusters = ['Cluster '+str(i) for i in range(7)]
docs = ['Doc '+str(i) for  i in range(txt.shape[0])]
bowdf = pd.DataFrame(np.round(trans,2) ,columns = clusters, index = docs)
bowdf['Cluster'] = np.argmax(bowdf.values,axis=1)
bowdf['Cluster']='Cluster ' + bowdf['Cluster'].astype(str)
bowdf
bow_clusters=bowdf['Cluster'].values.tolist()


# In[ ]:





# In[ ]:


#>>>>>Topic Modelling LDA using Bag of Words
bow = Pipeline([('cv',CountVectorizer()),('lda',LatentDirichletAllocation(n_components=7))])
bow_fit = bow.fit_transform(txt)



# In[ ]:


topics = ['Topic '+str(i) for i in range(bow['lda'].n_components)]
docs = ['Doc '+str(i) for  i in range(txt.shape[0])]
bowdf_topic = pd.DataFrame(np.round(bow_fit,2) ,columns = topics, index = docs)
bowdf_topic['Dominant topic'] = np.argmax(bowdf_topic.values,axis=1)
bowdf_topic['Dominant topic']='Topic ' + bowdf_topic['Dominant topic'].astype(str)
bowdf_topic



# In[ ]:


bowdf_topic['Cluster']=bow_clusters
bowdf_topic.to_csv('Bag_of_words.csv')


# In[ ]:





# In[ ]:


#>>>>> Using uni-grams and bi-grams
vectorizer = CountVectorizer(ngram_range=(1,2), min_df=20, max_features=5000)
vzx = vectorizer.fit_transform(txt)
vzx


# In[ ]:


# Principal Component Analysis
pca = PCA(0.85).fit_transform(vz.toarray())


# In[ ]:


# Kmeans Clustering
kmeans = KMeans(n_clusters=7).fit(pca)
trans=kmeans.transform(pca)
label = kmeans.predict(pca)


# In[ ]:


u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(pca[label == i , 0] , pca[label == i , 1] , label = i)
plt.legend()
plt.show()
plt.savefig("uni_bi_grams.png")


# In[ ]:


clusters = ['Cluster '+str(i) for i in range(7)]
docs = ['Doc '+str(i) for  i in range(txt.shape[0])]
ubdf = pd.DataFrame(np.round(trans,2) ,columns = clusters, index = docs)
ubdf['Cluster'] = np.argmax(ubdf.values,axis=1)
ubdf['Cluster']='Cluster ' + ubdf['Cluster'].astype(str)
ubdf


# In[ ]:


ub_clusters=ubdf['Cluster'].values.tolist()


# In[ ]:


#>>>>> Topic Modelling LDA with uni-grams and bi-grams
ub = Pipeline([('cv',CountVectorizer(ngram_range=(1,2), min_df=20, max_features=5000)),('lda',LatentDirichletAllocation(n_components=7))])
ub_fit = ub.fit_transform(txt)


# In[ ]:


topics = ['Topic '+str(i) for i in range(bow['lda'].n_components)]
docs = ['Doc '+str(i) for  i in range(txt.shape[0])]
ubdf = pd.DataFrame(np.round(ub_fit,2) ,columns = topics, index = docs)
ubdf['Dominant topic'] = np.argmax(ubdf.values,axis=1)
ubdf['Dominant topic']='Topic ' + ubdf['Dominant topic'].astype(str)
ubdf


# In[ ]:


ubdf['Cluster']=ub_clusters
ubdf.to_csv('uni_bi_grams.csv')


# In[ ]:





# In[ ]:





# In[ ]:


#>>>>>Word to Vector representation
def sen_vec(sentence,model):
    temp = np.zeros((len(sentence),1000),dtype=float)
    for i,j in enumerate(sentence):
        if j in model.wv.key_to_index.keys():
            temp[i,:] = model.wv.get_vector(j)
    return abs(np.sum(temp,axis=0)/len(sentence))




# In[ ]:


w2v = gensim.models.Word2Vec(txt, min_count= 7, vector_size= 1000, window=10)
vecs = np.array([sen_vec(x,w2v) for x in txt])
vecs


# In[ ]:


#>>>> Principal Component Analysis
pca = PCA(0.85).fit_transform(vecs)


# In[ ]:


#>>>>> Kmeans Clustering
kmeans = KMeans(n_clusters=7).fit(pca)
trans=kmeans.transform(pca)
label = kmeans.predict(pca)


# In[ ]:


u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(pca[label == i , 0] , pca[label == i , 1] , label = i)
plt.legend()
plt.show()
plt.savefig("W_2_V.png")


# In[ ]:


clusters = ['Cluster '+str(i) for i in range(7)]
docs = ['Doc '+str(i) for  i in range(txt.shape[0])]
w2vdf = pd.DataFrame(np.round(trans,2) ,columns = clusters, index = docs)
w2vdf['Cluster'] = np.argmax(w2vdf.values,axis=1)
w2vdf['Cluster']='Cluster ' + w2vdf['Cluster'].astype(str)
w2vdf


# In[ ]:


w2v_clusters=w2vdf['Cluster'].values.tolist()


# In[ ]:


#>>>>> Topic Modelling with Word to Vector
lda_model = LatentDirichletAllocation(n_components=7)             
topics = lda_model.fit_transform(vecs)


# In[ ]:


topic_names = ["Topic" + str(i) for i in range(lda_model.n_components)]
doc_names = ["Doc" + str(i) for i in range(txt.shape[0])]
w2v_document_topic = pd.DataFrame(np.round(topics, 2), columns=topic_names, index=doc_names)
topic = np.argmax(w2v_document_topic.values, axis=1)
w2v_document_topic['topic'] = topic
w2v_document_topic.head(7)


# In[ ]:


w2v_document_topic['Cluster']=w2v_clusters
w2v_document_topic.to_csv('W_2_V.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




