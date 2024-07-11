#-------------------------------------------------------------------------
# AUTHOR: Sidharth Basam
# FILENAME: indexing.py
# SPECIFICATION: Reads a CSV file and outputs the TF-IDF document-term matrix
# FOR: CS 4250- Assignment #1
# TIME SPENT: 0.45 hours
#-----------------------------------------------------------*/
# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas.
# You have to work here only with standard arrays.

# Importing some Python libraries
import csv
import math

# Reading the data in a csv file
documents = []
with open('collection.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            documents.append(row[0])

# Conducting stopword removal for pronouns/conjunctions (list)
stopWords = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and",
    "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", 
    "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", 
    "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
])

# Conducting stemming
stemming = {
    "cats": "cat",
    "dogs": "dog",
    "loves": "love",
}

def preprocess_text(text):
    words = text.lower().split()
    filtered_words = [stemming.get(word, word) for word in words if word not in stopWords]
    return filtered_words

processed_docs = [preprocess_text(doc) for doc in documents]

# Identifying the index terms
terms_set = set()
for doc in processed_docs:
    terms_set.update(doc)
terms = list(terms_set)

# Building the document-term matrix by using the tf-idf weights
def compute_tf(doc, terms):
    tf_dict = {term: 0 for term in terms}
    for word in doc:
        if word in tf_dict:
            tf_dict[word] += 1
    total_terms = len(doc)
    for term in tf_dict:
        tf_dict[term] /= total_terms
    return tf_dict

def compute_df(processed_docs, terms):
    df_dict = {term: 0 for term in terms}
    for doc in processed_docs:
        for term in set(doc):
            if term in df_dict:
                df_dict[term] += 1
    return df_dict

def compute_idf(df_dict, n_docs):
    idf_dict = {}
    for term, df in df_dict.items():
        idf_dict[term] = math.log10(n_docs / df)
    return idf_dict

def compute_tf_idf(tf, idf_dict):
    tf_idf = {}
    for word, tf_val in tf.items():
        tf_idf[word] = tf_val * idf_dict[word]
    return tf_idf

tf_list = [compute_tf(doc, terms) for doc in processed_docs]
df_dict = compute_df(processed_docs, terms)
idf_dict = compute_idf(df_dict, len(documents))
tf_idf_list = [compute_tf_idf(tf, idf_dict) for tf in tf_list]

# Document-term matrix
doc_term_matrix = [[tf_idf.get(term, 0) for term in terms] for tf_idf in tf_idf_list]

# Printing the document-term matrix
print("Document-Term Matrix (TF-IDF):")
print("Terms:", terms)
for row in doc_term_matrix:
    print(row)