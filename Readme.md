## INTRODUCTION 

Here the tf*idf scores are calculated, data is preprocessed and a posting file is. Using that information  the cosine similarity scores for the query are obtained which then give us a list of high-scoring documents.

I. Input

With weights -> python3 phase4.py input output W

retrieve edit 0.3 dog 0.0

Without weights -> python3 phase4.py input output

retrieve edit dog

II. Queries:
1. Preprocessing the query terms: The code has two options either we give the
query terms with their weights or we input only the query terms. If there are only query terms then we pre-process the query by removing the stopwords that are present in the stopwords.txt file as well as all numeric and special characters otherwise we extract we create a dictionary containing the key: value pair as follows {1:{edit:0.3,dog:0.0}}
2. We calculate the tf-idf scores for the query terms as we did in earlier phases.
The formula for tf calculations is :
Formula : tf(w,d) = count(w,d)/|D|
Here count(w,d) is the frequency of a word in a document and D is the total number of terms in that document.
The formula for idf calculations is :
Formula : idf(w) = log(|N|/wf(w))
Here N is the total number of documents which is 503 and wf(w) in the total number of times a word w occurs in the document. The log component is used to diminish the size for large values of C which in this case is 503.
Cosine similarity scores :
This code calculates the cosine similarity scores between queries and documents based on an inverted index and the calculated tf-idf scores for the query. It uses the formula for cosine similarity to calculate the relevance of each document to each query term, and then aggregates the scores across terms to produce an overall score for each document. Finally, it sorts the documents by their overall scores and stores the results in a dictionary for displaying the most relevant documents. Once all terms in the query have been processed, the doc_sim dictionary for the current query is sorted in descending order by the cosine similarity scores.

* The code begins by iterating over a dictionary containing the preprocessed query terms. A dictionary named doc_s is created to store the cosine similarity scores between the query and each document that contains at least one term from the query.

* For each term in the query, the code checks if the term exists in an inverted index named inverted_index. If the term exists in the index, the code retrieves the list of documents that contain the term and iterates over each document.

* For each document, the code retrieves a tf*idf score that represents the
relevance of the document to the current term. It then retrieves the length of the document vector named d_len and the score of the term in the query named query_s and the length of the query vector named q_len. Using these values, the code calculates the cosine similarity between the query and the document as (doc_s * query_s) / (d_len * q_len).

* The resulting cosine similarity score is added to the corresponding entry in the dictionary for the current document. If the document is not already in the dictionary, a new entry is created with the current cosine similarity score, and finally, the scores are sorted in descending order.

III. Sample query inputs
1. Input query: international affairs
2. Input query: computer network
3. Input query : identity theft
4. Input query : diet
 
