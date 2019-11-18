
# coding: utf-8

# In[2]:

import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm_notebook
from collections import Counter
import scipy
from scipy.sparse import *
from scipy import *

class HTZVectorizer():
    
    def __init__(self):
        self.vectorizer_bin = CountVectorizer(min_df=2, binary=True)
        self.vectorizer = CountVectorizer(min_df=2)
        
    
    def fit_transform(self, raw_documents, y=None):
        
        if isinstance(raw_documents, str):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")
            
        self.classes = set(y)
        
     #   X_cntvec = self.vectorizer.fit_transform(raw_documents) 
     #   X_idf = X_cntvec.toarray()
        X_idf = self.vectorizer_bin.fit_transform(raw_documents)
        
        idf_matrix = np.zeros((len(self.classes), X_idf.shape[1]))
        for i in tqdm_notebook(range(X_idf.shape[0])):
            idf_matrix[y[i]] += X_idf[i]
            
    #    del X_idf
            
        
        distrib_class = dict(Counter(y))
        values_class = np.zeros((len(distrib_class)))
        for key in sorted(distrib_class.keys()):
            values_class[key] = distrib_class[key]
            
        
        self.htz_matrix = np.zeros_like(idf_matrix)

        for i in tqdm_notebook(range(self.htz_matrix.shape[0])):
            without_one_class = X_idf.shape[0]-values_class[i]
            for j in range(self.htz_matrix.shape[1]):
                if idf_matrix[i][j] == 0:              #idf[i][j] - сколько раз слово j встретилось в классе i
                    IDF_wC = 1
                else:
                    IDF_wC = math.log(values_class[i]/idf_matrix[i][j], values_class[i])
                #    IDF_wC = math.log(values_class[i]/idf_matrix[i][j], 2)
                
                freq_all_doc = np.sum(idf_matrix[:,j]) - idf_matrix[i][j]
                
                
         #       print(math.log(without_one_class/freq_all_doc, 2))
         #       print(IDF_wC)
                
                if freq_all_doc == 0:
                    IDF_Dc = 1
                else:
                    IDF_Dc = math.log(without_one_class/freq_all_doc, without_one_class)
                self.htz_matrix[i][j] = IDF_Dc - IDF_wC
                
            #    self.htz_matrix[i][j] = (math.log(without_one_class/freq_all_doc, 2)- IDF_wC)/ np.maximum(math.log(without_one_class/freq_all_doc, 2), IDF_wC)
                if self.htz_matrix[i][j] < 0:
                    self.htz_matrix[i][j] = 0
                           
        docs_len = np.array([len(raw_documents.iloc[i].split()) for i in range(len(raw_documents))])
        diag_lens = diags(1/np.log(docs_len +1e-10))
   #     diag_lens = diags(1/(docs_len))

        X_cntvec = self.vectorizer.fit_transform(raw_documents)
        X_cntvec.data = np.log(X_cntvec.data + 1)
        
        tf_matrix = diag_lens @ X_cntvec

               
    #    RIC_matrix = np.zeros((X_cntvec.shape[0], len(self.classes)))
        RIC_matrix = tf_matrix @ self.htz_matrix.T
        
  #      for i in tqdm_notebook(range(RIC_matrix.shape[0])):
   #         for j in range(RIC_matrix.shape[1]):         
    #            if np.sum(RIC_matrix[i]) - RIC_matrix[i][j] != 0:
     #               RIC_matrix[i][j] = RIC_matrix[i][j]/(np.sum(RIC_matrix[i]) - RIC_matrix[i][j])
        
        return scipy.sparse.csr_matrix(RIC_matrix)
    
    def transform(self, raw_documents):
        X_cntvec = self.vectorizer.transform(raw_documents)
        docs_len = np.array([len(raw_documents.iloc[i].split()) for i in range(len(raw_documents))])
        diag_lens = diags(1/np.log(docs_len + 1e-10))
     #   diag_lens = diags(1/(docs_len))
        
        X_cntvec.data = np.log(X_cntvec.data + 1)
        tf_matrix = diag_lens @ X_cntvec
        
     #   RIC_matrix = np.zeros((X_cntvec.shape[0], len(self.classes)))        
        RIC_matrix = tf_matrix @ self.htz_matrix.T
        
    #    for i in tqdm_notebook(range(RIC_matrix.shape[0])):
   #         for j in range(RIC_matrix.shape[1]):
     #           if np.sum(RIC_matrix[i]) - RIC_matrix[i][j] != 0:
      #              RIC_matrix[i][j] = RIC_matrix[i][j]/(np.sum(RIC_matrix[i]) - RIC_matrix[i][j])
        
        return scipy.sparse.csr_matrix(RIC_matrix)
        
            

