#!/usr/bin/env python
# coding: utf-8

# In[291]:


import pandas as pd
import os
import numpy as np
import re
import sys
from pandas import DataFrame
from sklearn.utils import shuffle
from collections import OrderedDict
import hashlib


# In[292]:


APACHE_ACCESS_LOG_PATTERN = '^(\S+) (\S+) (\S+) (\S+) (\S+) (.*)'


# In[293]:


def parse_hdfs_file(file):
    ab = []
    file_line = file.readlines()
    for line in range(len(file_line)):
        match = re.findall(APACHE_ACCESS_LOG_PATTERN, file_line[line])
        for mat in range(len(match)):
            bd = re.sub("[\d.-]+", "<*>", match[mat][5])
            eventid = hashlib.md5(' '.join(bd).encode('utf-8')).hexdigest()[0:8]
            ab.append([match[mat][0],match[mat][1],match[mat][2],match[mat][3],match[mat][4],match[mat][5],bd, eventid] )
    return ab 
    


# In[294]:


fo = open("HDFS.log", "r")    


# In[295]:


dta = parse_hdfs_file(fo)


# In[296]:


data = pd.DataFrame(dta,columns=["Date","Time","Pid","Info","Component","Content","EventTemplate","EventId"])


# In[297]:


data.head(10)


# In[298]:


def read_file(data):
    data_dict = OrderedDict()
    for idx, row in data.iterrows():
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            if not blk_Id in data_dict:
                data_dict[blk_Id] = []
            data_dict[blk_Id].append(row['EventId'])
    data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
    
    return data_df


# In[299]:


df_train_append = read_file(data)


# In[300]:


df_train_append.head(10)


# In[301]:


label_csv = pd.read_csv('anomaly_label.csv')


# In[302]:


label_csv.head(10)


# In[303]:


label_data =label_csv.set_index('BlockId')
label_dict = label_data['Label'].to_dict()
label_dict


# In[304]:


df_train_append


# In[305]:


df_train_append['Label'] = df_train_append['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)


# In[306]:


df_event = pd.DataFrame(df_train_append)


# In[307]:


df_event.head(10)


# In[308]:


from sklearn.model_selection import train_test_split


# In[309]:


x_train, x_test, y_train, y_test = train_test_split(df_train_append['EventSequence'].values, df_train_append['Label'].values, test_size = 0.2) 


# In[310]:


from collections import Counter
from scipy.special import expit
from itertools import compress


# In[311]:


class FeatureExtractor(object):

    def __init__(self):
        self.idf_vec = None
        self.mean_vec = None
        self.events = None
        self.term_weighting = None
        self.normalization = None
        self.oov = None

    def fit_transform(self, X_seq, term_weighting=None, normalization=None, oov=False, min_count=1):
        """ Fit and transform the data matrix
        Arguments
        ---------
            X_seq: ndarray, log sequences matrix
            term_weighting: None or `tf-idf`
            normalization: None or `zero-mean`
            oov: bool, whether to use OOV event
            min_count: int, the minimal occurrence of events (default 0), only valid when oov=True.
        Returns
        -------
            X_new: The transformed data matrix
        """
        print('====== Transformed train data summary ======')
        self.term_weighting = term_weighting
        self.normalization = normalization
        self.oov = oov

        X_counts = []
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)
        self.events = X_df.columns
        X = X_df.values
        if self.oov:
            oov_vec = np.zeros(X.shape[0])
            if min_count > 1:
                idx = np.sum(X > 0, axis=0) >= min_count
                oov_vec = np.sum(X[:, ~idx] > 0, axis=1)
                X = X[:, idx]
                self.events = np.array(X_df.columns)[idx].tolist()
            X = np.hstack([X, oov_vec.reshape(X.shape[0], 1)])
        
        num_instance, num_event = X.shape
        if self.term_weighting == 'tf-idf':
            df_vec = np.sum(X > 0, axis=0)
            self.idf_vec = np.log(num_instance / (df_vec + 1e-8))
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1)) 
            X = idf_matrix
        if self.normalization == 'zero-mean':
            mean_vec = X.mean(axis=0)
            self.mean_vec = mean_vec.reshape(1, num_event)
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        elif self.normalization == 'sigmoid':
            X[X != 0] = expit(X[X != 0])
        X_new = X
        
        print('Train data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1])) 
        return X_new
    
    def transform(self, X_seq):
        """ Transform the data matrix with trained parameters
        Arguments
        ---------
            X: log sequences matrix
            term_weighting: None or `tf-idf`
        Returns
        -------
            X_new: The transformed data matrix
        """
        print('====== Transformed test data summary ======')
        X_counts = []
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)
        empty_events = set(self.events) - set(X_df.columns)
        for event in empty_events:
            X_df[event] = [0] * len(X_df)
        X = X_df[self.events].values
        if self.oov:
            oov_vec = np.sum(X_df[X_df.columns.difference(self.events)].values > 0, axis=1)
            X = np.hstack([X, oov_vec.reshape(X.shape[0], 1)])
        
        num_instance, num_event = X.shape
        if self.term_weighting == 'tf-idf':
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1)) 
            X = idf_matrix
        if self.normalization == 'zero-mean':
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        elif self.normalization == 'sigmoid':
            X[X != 0] = expit(X[X != 0])
        X_new = X

        print('Test data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1])) 

        return X_new


# In[ ]:





# In[312]:


feature_extractor = FeatureExtractor()
x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
x_test = feature_extractor.transform(x_test)


# In[313]:


x_train


# In[314]:


from sklearn.ensemble import RandomForestClassifier

rfclf = RandomForestClassifier(max_depth=1, random_state=42)
rfclf = rfclf.fit(x_train, y_train)

from sklearn.metrics import f1_score

predicted_y = rfclf.predict(x_test)
print("test accuracy: ",f1_score(y_test, predicted_y, average='micro'))


# In[315]:


predicted_y


# In[316]:


from sklearn.externals import joblib  

joblib.dump(rfclf, "trained-model.pkl")


# In[ ]:





# In[ ]:




