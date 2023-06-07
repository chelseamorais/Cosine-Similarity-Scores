import os
import string
import re
import time
import nltk
import itertools
import math
import operator
import glob
import sys
import html2text
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

def Removing_Stopwords(text):
    stop_words = [word for word in open('Stopwords.txt','r').read().split('\n')]
    text = text.replace("\n","").lower()
    text = re.sub(pattern = "[^\w\s]", repl="",string = text)   #Removing all punctuations
    text = re.sub(pattern = "[0-9]*", repl="",string = text)    #Removing all numbers
    text = re.sub(pattern = "_*", repl="",string = text)
    tokens =word_tokenize(text)
    tokens1 = [token for token in tokens if token not in stop_words and len(token) > 2]
    return tokens1

def read_data(path):

    st = time.time()
    timer = []
    doc_time_calc = [10, 20, 40, 80, 100, 200, 300, 400, 500]
    contents = []
    current_path = os.getcwd()
    filepath = current_path+"/"+path+"/*.html"
    files=glob.glob(filepath)
    for filename in files:
        f=open(filename, 'rb')
        text = html2text.html2text(f.read().decode(errors='replace'))
        pattern = r'(?<=/)\d{3}(?=\.)'
        match = re.search(pattern, filename)
        if match:
            filename = match.group(0)

        #filename = re.sub('\D',"",filename)
        #print(filename)
        contents.append((int(filename),text))
        if int(filename) in doc_time_calc:
            t = time.time() - st
            timer.append(t)
    return contents,timer

def vocab(data):
    tokens = []
    for tn in data.values():
        tokens += tn
    fdist = FreqDist(tokens)
    #len1 = {key: val for key,val in fdist.items() if val == 1}
    new_dict = {key: val for key,val in fdist.items() if val != 1}
    return list(new_dict.keys())

def preprocess(contents):
    dataDict = {}
    count = 0
    st = time.time()
    timer1 = []
    doc_time_calc = [10, 20, 40, 80, 100, 200, 300, 400, 500]
    for content in contents:
        count +=1
        tokens = Removing_Stopwords(content[1]) # REMOVE PUNCTUATIONS, NUMBERS, WORDS OF LENGTH 1
        dataDict[content[0]] = tokens
        #print(content[0])
        if count in doc_time_calc:
            t = time.time() - st
            timer1.append(t)
    #print(dataDict)
    #print("\n\n\n\n\n\n")
    return dataDict,timer1

def tf(tokens):
    tf_score = {}
    for token in tokens:
        tf_score[token] = tokens.count(token)/len(tokens) #Normalizing by number of documents
        #print(tf_score[token])
    #print("\n\n\n\n\n\n\n\n\n\n\n")
    return tf_score

def idf(data):
    idf_score = {}
    Nofdocs = len(data)
    #print(len(data))
    all = vocab(data)
    for w in all:
        wcount = 0
        for lists in data.values():
            if w in lists:
                wcount += 1
        idf_score[w] = math.log10(Nofdocs/wcount) # log for big values of noof documents
    return idf_score

def tfidf(data, idf_s):
    scores = {}
    for key,value in data.items():
        scores[key] = tf(value)
    for doc,tf_s in scores.items():
        for token, score in tf_s.items():
            tf_1 = score              #Calculated tf score
            idf = idf_s[token]  #Calculates idf score
            tf_s[token] = tf_1 * idf
    return scores

def inverted_index1(data):
    all = vocab(data)
    #print(len(all_words))
    val = {}
    for w in all:
        for doc, tokens in data.items():
            if w in tokens :
                if w in val.keys():
                    val[w].append(doc)
                else:
                    val[w] = [doc]
    #print(index)
    return val

def outfile(scores):
    current_path = os.getcwd()
    #print(current_path)
    #print(scores)
    for doc in scores.keys():
        print(doc)
        filepath = current_path+"/"+sys.argv[2]+"/"+str(doc)+".wts"

        nf = open(filepath,"w+")
        for i ,j in scores[doc].items():
            nf.write(i)
            nf.write("\t\t\t %.8f\n" % (j,))
            
        nf.close()
def queries_tfidf(query, idf_s):
    dict = {}
    for key, value in query.items():
        dict[key] = tf(value)
    for key, tf_s in dict.items():
        for token, s in tf_s.items():
            idf = 0
            tf_calc = s
            if token in idf_s.keys():
                idf = idf_s[token]
            tf_s[token] = tf_calc * idf
    return dict
    
def newoutfile(my_dict):
    token_weights = {}
    current_path = os.getcwd()
    
# Iterate over the documents in the dictionary
    for doc, tokens in my_dict.items():
        # Iterate over the tokens in each document
        for token, weight in tokens.items():
            # Add the token weight to the dictionary
            if token in token_weights:
                token_weights[token].append((doc, weight))
            else:
                token_weights[token] = [(doc, weight)]

# Iterate over the tokens and their weights and print them
    count = 0
    freq = 0
    filepath = current_path+"/"+sys.argv[2]+"/"+str("Dictionary File")+".wts"
    nf = open(filepath,"w+")
    a = "Token"
    b = "Nos of Docs"
    c = "Posting File Location"
    row = '{:<18}{:<18}{:<12}'.format(a,b,c)
    nf.write(row + '\n')
    for token, weights in token_weights.items():
        #print(token)
        row = '{:<18}'.format(token)
        nf.write(row)
        for doc, weight in weights:
            freq+=1
            count+=1
        row = '{:<18}{:<12}'.format(str(freq),count)
        nf.write(row + '\n')
        freq =0
    
    filepath = current_path+"/"+sys.argv[2]+"/"+str("Postings File")+".wts"
    newf = open(filepath,"w+")
    a = "Doc ID"
    b = "Token weight"
    row = '{:<12}{}'.format(a,b)
    newf.write(row + '\n')
    for token, weights in token_weights.items():
        for doc, weight in weights:
            row = '{:<15}{}'.format(str(doc),weight)
            newf.write(row + '\n')
    
    


start = time.time()
#main method
args = sys.argv
data,timer = read_data(args[1])
data1,timer1 = preprocess(data)

a = vocab(data1)
# remove words of frequency 1
for i,j in data1.items():
    new_words = [word for word in j if word in a]
    data1[i]=new_words

inverted_index = inverted_index1(data1)
idf_scores = idf(data1)
scores = tfidf(data1,idf_scores)
#newoutfile(scores)
#print(scores)
#outfile(scores)

query = input("retrieve")

if len(args) == 4:
    if args[3] == 'W':

        query_scores = {1: {query[i]: query[i+1] for i in range(0, len(query)-1, 2)}}
        queries = {1 : list(query_scores[1].keys())}
        #queries = preprocess_queries(query)
        
else:
    queries = {}
    t = Removing_Stopwords(query)
    queries[1] = t
    #print(queries)
    query_scores = queries_tfidf(queries,idf_scores)
    #print(query_scores)


query_docs = {}
for key, value in queries.items():
    doc_sim = {}
    for term in value:
        if term in inverted_index.keys():
            docs = inverted_index[term]
            for doc in docs:
                doc_s = scores[doc][term]
                d_len = math.sqrt(sum(x ** 2 for x in scores[doc].values()))
                query_s = query_scores[key][term]
                q_len = math.sqrt(sum(x ** 2 for x in query_scores[key].values()))
                cos_sim = (doc_s * query_s) / (d_len * q_len)
                if doc in doc_sim.keys():
                    doc_sim[doc] += cos_sim
                else:
                    doc_sim[doc] = cos_sim
    ranked = sorted(doc_sim.items(), key=operator.itemgetter(1), reverse=True)
    query_docs[key] = ranked



print("Top scoring documents")

for i in range(1, len(query_docs) + 1):
    docs = query_docs[i][:10]
    for i in docs:
        print(i)

    
