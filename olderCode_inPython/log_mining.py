#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np
import re
import sys
from pandas import DataFrame
from sklearn.utils import shuffle
from collections import OrderedDict
import hashlib


# In[2]:


from Extractor import FeatureExtractor


# In[3]:


# APACHE_ACCESS_LOG_PATTERN = '^(\S+) (\S+) (\S+) (\S+) (\S+) (.*)'


# In[4]:


def parse_hdfs_file(file_ab, APACHE_ACCESS_LOG_PATTERN):
    ab = []
    file_line = file_ab.readlines()
    for line in range(len(file_line)):
        match = re.findall(APACHE_ACCESS_LOG_PATTERN, file_line[line])
        for mat in range(len(match)):
            bd = re.sub("[\d.-]+", "<*>", match[mat][5])
            eventid = hashlib.md5(' '.join(bd).encode('utf-8')).hexdigest()[0:8]
            ab.append([match[mat][0],match[mat][1],match[mat][2],match[mat][3],match[mat][4],match[mat][5],bd, eventid] )
    return ab 
    


# In[5]:


# fo = open("HDFS.log", "r")    


# In[6]:


# dta = parse_hdfs_file(fo)


# In[7]:


# data = pd.DataFrame(dta,columns=["Date","Time","Pid","Info","Component","Content","EventTemplate","EventId"])


# In[8]:


# data.head(10)


# In[9]:


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


# In[10]:


def dict_label(df_train_append, label_file):
        
    label_data =label_file.set_index('BlockId')
    label_dict = label_data['Label'].to_dict() 
    df_train_append['Label'] = df_train_append['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)
    return pd.DataFrame(df_train_append)


# In[11]:


# df_train_append = read_file(data)


# In[12]:


# df_train_append.head(10)


# In[13]:


# label_csv = pd.read_csv('anomaly_label.csv')


# In[14]:


# label_csv.head(10)


# In[15]:


# label_data =label_csv.set_index('BlockId')
# label_dict = label_data['Label'].to_dict()
# label_dict


# In[16]:


# df_train_append


# In[17]:


# df_train_append['Label'] = df_train_append['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)


# In[18]:


# df_event = pd.DataFrame(df_train_append)


# In[19]:


# df_event.head(10)


# In[20]:


# from sklearn.model_selection import train_test_split


# In[21]:


# x_train, x_test, y_train, y_test = train_test_split(df_train_append['EventSequence'].values, df_train_append['Label'].values, test_size = 0.2) 


# In[22]:


# from collections import Counter
# from scipy.special import expit
# from itertools import compress


# In[23]:


# feature_extractor = FeatureExtractor()
# x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
# x_test = feature_extractor.transform(x_test)


# In[24]:


# from sklearn.ensemble import RandomForestClassifier

# rfclf = RandomForestClassifier(max_depth=1, random_state=42)
# rfclf = rfclf.fit(x_train, y_train)

# from sklearn.metrics import f1_score

# predicted_y = rfclf.predict(x_test)
# print("test accuracy: ",f1_score(y_test, predicted_y, average='micro'))


# In[25]:


# predicted_y


# In[26]:


# from sklearn.externals import joblib  

# joblib.dump(rfclf, "trained-model.pkl")


# In[ ]:





# In[ ]:





# In[ ]:




