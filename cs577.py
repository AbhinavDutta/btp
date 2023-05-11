#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sys
import numpy as np
import pandas as pd
import mmh3
import random
import solcx
solcx.install_solc('0.4.22')  # install a specific version of the Solidity compiler
solcx.set_solc_version('0.4.22')


# In[59]:


def hash_kmer(kmer):
    # calculate murmurhash using a hash seed of 42
    hash = mmh3.hash64(kmer, 42)[0]
    if hash < 0: hash += 2**64
    return hash

def build_kmers(sequence, ksize):
    kmers = []
    n_kmers = len(sequence) - ksize + 1

    for i in range(n_kmers):
        kmer = sequence[i:i + ksize]
        kmers.append(hash_kmer(kmer))

    return kmers

def hamming_distance(a,b):
    cnt=0
    for i in range(0,25):
        if a[i] != b[i]:
            cnt+=1
    return cnt/25



def jaccard_similarity(a, b):
    intersection = len(a.intersection(b))
    union = len(a.union(b))
    
    return intersection / union

def lsh(kmers):
    return kmers[kmers.index(min(kmers))]


# In[ ]:





# In[18]:


data = pd.read_csv('s2.csv',dtype=str,header=0)
data.head()
len(data)


# In[23]:


data = pd.read_csv('btp/61k.csv',dtype=str,header=0)
len(data)


# In[81]:


i=0
dict_code_kmers={}
code=[]
label=[]
dirtycode=[]
bytecode=[]

for index,row in data.iterrows():
    if(i>100):
        break
    dirty_code = row['code']
    if 'pragma solidity ^0.4.22' not in dirty_code:
        continue
    dirtycode.append(dirty_code)
    tokens = dirty_code.split()
    clean_code = ""
    for t in tokens:
        clean_code = clean_code+t
    if (len(clean_code)<100):
        continue
        
        
    print(i)
    try:
        compiled_contract = solcx.compile_source(dirty_code)
    except Exception as e:
        #print(e)
        continue
        
    
    for keyk, valv in compiled_contract.items():
        bc = ''
        if 'bin-runtime' in valv:
            bc = valv['bin-runtime']
        elif 'bin' in valv:
            bc = valv['bin']
        else:
            print(compiled_contract)
            break
        if(len(bc)==0):
            break
        bytecode.append(bc)
        code.append(clean_code)
        label.append(row['label'])
        i=i+1
            
        break
   
        
    
   
n=i
i=0
for i in range(0,n):
    kmer_s = build_kmers(code[i],4)
    if(len(kmer_s)>1200):
        kmer_s = ( random.sample(kmer_s, 1200))
    dict_code_kmers[i]= set(kmer_s)
    
    


# In[85]:



for i in range(0,len(bytecode)):
    kmer_s = build_kmers(bytecode[i],4)
    if(len(kmer_s)>1200):
        kmer_s = ( random.sample(kmer_s, 1200))
    dict_code_kmers[i]= set(kmer_s)
    


# In[86]:


jacc = np.zeros(n*n)
jacc = jacc.reshape(n,n)

hamm = np.zeros(n*n)
hamm = hamm.reshape(n,n)
n


# In[87]:



import editdistance
for j in range(0,n):
    for k in range(0,n):
        jacc[j][k] = 1-jaccard_similarity(dict_code_kmers[j],dict_code_kmers[k])
        #jacc[j][k] = editdistance.eval(code[j],code[k])
        #jacc[j][k] = editdistance.eval(bytecode[j],bytecode[k])
        print(j,k)
        hamm[j][k] = hamming_distance(label[j],label[k])
        
jacc


# In[88]:


from sklearn.cluster import AgglomerativeClustering 
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

'''
X = np.array(jacc)
clustering = AgglomerativeClustering(linkage='average',affinity="precomputed").fit(X)

clustering.labels_
'''
k_arr = []
silhouette_scores = [] 

for n_cluster in range(2, 20):
    k_arr.append(n_cluster)
    silhouette_scores.append(silhouette_score(jacc,AgglomerativeClustering(n_clusters = n_cluster,linkage='single',affinity='precomputed').fit_predict(jacc))) 
    


# In[89]:


# Plotting a bar graph to compare the results 
from matplotlib import pyplot as plt
plt.bar(k_arr, silhouette_scores) 
plt.xlabel('Number of clusters', fontsize = 10) 
plt.ylabel('Silhouette Score', fontsize = 10) 
plt.show() 


# In[91]:


print(silhouette_scores)


# In[96]:


X = np.array(jacc)
clustering = AgglomerativeClustering(n_clusters=2,linkage='single',affinity="precomputed").fit(X)

clustering.labels_


# In[98]:


plt.hist(clustering.labels_, bins='auto')


# In[12]:


import ssdeep
ctph = np.zeros(n*n)
ctph = ctph.reshape(n,n)
for j in range(0,n):
    for k in range(0,n):
        h1=ssdeep.hash(code[j])
        h2=ssdeep.hash(code[k])
        ctph[j][k] = ssdeep.compare(h1,h2)


# In[71]:


import scipy.stats
x= hamm.flatten()
y= jacc.flatten()

result = scipy.stats.spearmanr(x, y)
print(result)
r,p = scipy.stats.pearsonr(x, y)
print(r,p)


# In[ ]:





# In[83]:


import scipy.stats

k_val=[]
cor_val=[]
for k_len in range(1,20):
    for i in range(0,len(code)):
        kmer_s = build_kmers(bytecode[i],k_len)
        if(len(kmer_s)>1200):
            kmer_s = ( random.sample(kmer_s, 1200))
        dict_code_kmers[i]= set(kmer_s)
    n=len(code)
    jacc = np.zeros(n*n)
    jacc = jacc.reshape(n,n)

    hamm = np.zeros(n*n)
    hamm = hamm.reshape(n,n)
    
    for j in range(0,n):
        for k in range(0,n):
            jacc[j][k] = 1-jaccard_similarity(dict_code_kmers[j],dict_code_kmers[k])
            hamm[j][k] = hamming_distance(label[j],label[k])
    x= hamm.flatten()
    y= jacc.flatten()
    r,p = scipy.stats.pearsonr(x, y)
    print(r,p)
    k_val.append(k_len)
    cor_val.append(r)
    


# In[32]:





# In[84]:


plt.plot(k_val, cor_val)
plt.xlabel('Size of n (in n-gram)', fontsize = 10) 
plt.ylabel('Correlation', fontsize = 10) 
plt.show()


# In[31]:


f = open("demo.sol", "r")
demo_str = f.read()


# In[30]:



# compile the contract and get the bytecode
solcx.install_solc('0.8.3')  # install a specific version of the Solidity compiler
contract_source_code = """
pragma solidity ^0.8.0;

contract MyContract {
    uint256 public myVariable;

    constructor() {
        myVariable = 0;
    }

    function setVariable(uint256 newValue) public {
        myVariable = newValue;
    }
}
"""

compiled_contract = solcx.compile_source(contract_source_code)
#print(compiled_contract)

#bytecode = compiled_contract['contracts']['<stdin>:MyContract']['evm']['bytecode']['object']
bytecode = compiled_contract['<stdin>:MyContract']['bin-runtime']
# convert the bytecode to a list of integers
bytecode_list = [int(bytecode[i:i+2], 16) for i in range(2, len(bytecode), 2)]
# convert the list to a numpy array
bytecode_array = np.array(bytecode_list)
print(bytecode_array)


# In[38]:


data['code'][0]
data


# In[14]:


import solcx
#import numpy as np

# compile the contract and get the bytecode

#solcx.install_solc('0.4.22')  # install a specific version of the Solidity compiler

compiled_contract = solcx.compile_source(data['code'][0])
#print(compiled_contract)

#bytecode = compiled_contract['contracts']['<stdin>:MyContract']['evm']['bytecode']['object']
bytecode = compiled_contract['<stdin>:Greeter']['bin-runtime']
# convert the bytecode to a list of integers
bytecode_list = [int(bytecode[i:i+2], 16) for i in range(2, len(bytecode), 2)]
# convert the list to a numpy array
bytecode_array = np.array(bytecode_list)
print(len(bytecode_array))


# In[26]:


solcx.install_solc('0.4.22')  # install a specific version of the Solidity compiler
solcx.set_solc_version('0.4.22')
solcx.get_solc_version(True)
print(data['code'][0])


# In[20]:


import solcx
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# compile the contract and get the source code
contract_source_code = data['code'][0]
compiled_contract = solcx.compile_source(contract_source_code)
#print(compiled_contract)

#source_code = compiled_contract['contracts']['<stdin>:MyContract']['source']

# tokenize the source code using NLTK
tokens = nltk.word_tokenize(contract_source_code)

# convert the tokens to a bag-of-words representation using sklearn
vectorizer = CountVectorizer()
bow = vectorizer.fit_transform([' '.join(tokens)]).toarray()

# convert the bag-of-words to a numpy array
bow_array = np.array(bow[0])
print(bow)
print(bow_array)


# In[25]:


import solcx
import networkx as nx

# Set the Solidity compiler version

# Read the Solidity source code from the file


# Compile the Solidity source code
source_code = data['code'][0]
compiled_sol = solcx.compile_source(source_code)

# Get the AST of the contract
ast = compiled_sol['<stdin>:Greeter']['ast']
print(ast)
# Build a control flow graph from the AST
g = nx.DiGraph()
for node in ast['children']:
    if node['nodeType'] == 'FunctionDefinition':
        function_name = node['name']
        function_body = node['body']['statements']
        for i, statement in enumerate(function_body):
            if statement['nodeType'] == 'ExpressionStatement' and statement['expression']['nodeType'] == 'FunctionCall':
                callee = statement['expression']['expression']['referencedDeclaration']
                g.add_edge(function_name, callee)

# Extract features from the control flow graph
in_degrees = dict(g.in_degree)
out_degrees = dict(g.out_degree)
number_of_nodes = len(g.nodes)
number_of_edges = len(g.edges)

# Convert the features into a vector representation
feature_vector = [in_degrees, out_degrees, number_of_nodes, number_of_edges]

print(feature_vector)


# In[28]:


from gensim.models import KeyedVectors as word2vec
vectors_text_path = '/home/abhinav/Desktop/btp/code/btp/token_vecs.txt'
model = word2vec.load_word2vec_format(vectors_text_path, binary=False)


# In[35]:


from gensim.models import Word2Vec
model_ted = Word2Vec(sentences=sentences_ted, size=100, window=5, min_count=5, workers=4, sg=0)


# In[37]:


import solcx
import networkx as nx

# Set the Solidity compiler version

# Read the Solidity source code from the file
source_code = data['code'][0]
# Compile the Solidity source code
compiled_sol = solcx.compile_source(source_code)

# Get the AST of the contract
ast = compiled_sol['<stdin>:MyContract']['ast']

# Build a control flow graph from the AST
g = nx.DiGraph()
for node in ast['children']:
    if node['nodeType'] == 'FunctionDefinition':
        function_name = node['name']
        function_body = node['body']['statements']
        for i, statement in enumerate(function_body):
            if statement['nodeType'] == 'ExpressionStatement' and statement['expression']['nodeType'] == 'FunctionCall':
                callee = statement['expression']['expression']['referencedDeclaration']
                g.add_edge(function_name, callee)

# Extract features from the control flow graph
in_degrees = dict(g.in_degree)
out_degrees = dict(g.out_degree)
number_of_nodes = len(g.nodes)
number_of_edges = len(g.edges)

# Convert the features into a vector representation
feature_vector = [in_degrees, out_degrees, number_of_nodes, number_of_edges]

print(feature_vector)


# In[ ]:




