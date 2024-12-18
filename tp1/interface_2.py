
import streamlit as st
import pandas as pd
import nltk
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
import ast




import nltk
import ast
import numpy as np
import pandas as pd  # Ensure pandas is imported
import numpy as np
import nltk
import ast
nltk.download('stopwords')
import re

    # Open file for tokenized query processing
import nltk
import numpy as np
def evaluate_relevance(token_value, words, stemmer_value):
 

    # Define stopwords to keep
    keep_stopwords = {'and', 'not', 'or'}

    # Get the default stopwords list and remove 'and', 'not', 'or'
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords -= keep_stopwords

    # Initialize the stemmer and process the words
    if stemmer_value == "Porter":
        stemmer = nltk.PorterStemmer()
    elif stemmer_value == "Lancaster":
        stemmer = nltk.LancasterStemmer()
    else:
        raise ValueError("Invalid stemmer value. Choose 'Porter' or 'Lancaster'.")

    # Stem words and filter out stopwords
    words = [stemmer.stem(word) for word in words if word.lower() not in stopwords]
    print("Processed Words:", words)
    valid_operators = {'and', 'or', 'not'}

    for i, word in enumerate(words):
        if i!=0 and (words[i-1].lower() not in valid_operators) and( word.lower() not in valid_operators):
             raise ValueError(f"Invalid sequence: two terms '{words[i -1]}' and '{word}' cannot follow each other directly. Operators are missing.")

        if word.lower() in valid_operators:
            # Check for invalid sequences: AND OR, OR AND, NOT AND, etc.
            
            if( i == 0 and word in  {'and', 'or'} )or i == len(words) - 1 : 
                # Operator cannot be the first or last element
                raise ValueError(f"Invalid position for operator '{word}'. Operators cannot be at the beginning or end.")
            elif words[i - 1].lower() in valid_operators  and words[i ].lower() in   {'and', 'or'} or words[i-1].lower() in {'not'}  and words[i].lower() in {'not'}   :  # Two consecutive operators are not allowed
                raise ValueError(f"Invalid sequence: '{words[i - 1]}' followed by '{word}'.")
            # elif i + 1 < len(words) and words[i + 1].lower() in valid_operators:  # Operator cannot be followed by another operator
            #     raise ValueError(f"Invalid sequence: '{word}' followed by '{words[i + 1]}'. Operators cannot follow each other.")
            # if word.lower() == 'not' and (i + 1 == len(words) or words[i + 1].lower() in valid_operators):
            #    raise ValueError(f"Invalid sequence: 'not' cannot be followed by another operator or the end of the sentence.")
            elif (words[i+1].lower() not in valid_operators) and( word.lower() not in valid_operators):
             raise ValueError(f"Invalid sequence: two terms '{words[i - 1]}' and '{word}' cannot follow each other directly. Operators are missing.")
            
    # # Initialize a matrix to store term-document relevance
    t = np.zeros((len(words), 6), dtype=object)

    # Map logical operators to markers
    for i, word in enumerate(words):
        if word.lower() in keep_stopwords:
            t[i, :] = word.lower()

    # Open the file containing document data
    try:
        with open(f'description{token_value}{stemmer_value}.txt', 'r') as f:
            for i, word in enumerate(words):
                f.seek(0)  # Ensure we start reading from the beginning of the file
                for line in f:
                    components = line.strip().split()
                    doc_id = int(components[0])  # Document ID (assumed first element)
                    term = components[1]         # Term (assumed second element)

                    # If the term matches the current query word, update the matrix
                    if term.lower() == word.lower():
                        t[i, doc_id - 1] = 1  # Set the relevant document column to 1
    except FileNotFoundError:
        print(f"File description{token_value}{stemmer_value}.txt not found.")
        return []

    # Print the term-document matrix
    print("Term-Document Matrix:\n", t)

    # Evaluate relevance scores for each document
    st.write("valide")
    final = []
    # for doc_id in range(6):
    #     # Convert column vector into a boolean list or keep operators
    #     bool_vector = [
    #         True if x == 1 else False if x == 0 else x
    #         for x in t[:, doc_id]
    #     ]

    #     # Evaluate boolean logic using a stack
    #     stack = []
    #     i = 0
    #     while i < len(bool_vector):
    #         elem = bool_vector[i]
    #         if elem == 'and':
    #             operand1 = stack.pop() if stack else False
    #             operand2 = bool_vector[i + 1] if i + 1 < len(bool_vector) else False
    #             stack.append(operand1 and operand2)
    #             i += 1
    #         elif elem == 'or':
    #             operand1 = stack.pop() if stack else False
    #             operand2 = bool_vector[i + 1] if i + 1 < len(bool_vector) else False
    #             stack.append(operand1 or operand2)
    #             i += 1
    #         elif elem == 'not':
    #             operand = bool_vector[i + 1] if i + 1 < len(bool_vector) else False
    #             stack.append(not operand)
    #             i += 1
    #         else:
    #             stack.append(elem)
    #         i += 1

    #     # Get the final evaluation (convert True/False to 1/0)
    #     if stack and isinstance(stack[0], bool):  # Ensure the stack's first element is boolean
    #         relevance_score = int(stack[0])
    #     else:
    #         relevance_score = 0
    #     final.append((doc_id + 1, relevance_score))

    return final


def search_action(query_value, token_value, stemmer_value, index_selection, matching_selection,k,b):
    final_list = []
    size = 0
    voc = 0
    
    if matching_selection == "without":
        # Load data from the corresponding file
        with open(f'description{token_value}{stemmer_value}.txt', 'r') as f:
            ExpReg = nltk.RegexpTokenizer(r"(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,\-]\d+)?%?|\w+(?:[\-/]\w+)*")
            for line in f:
                components = line.strip().split()
                if len(components) > 3:
                    doc_id = components[0]
                    term = components[1]
                    freq = components[2]
                    weight = components[3]
                    # Extract remaining components as positions (in an array)
                    pos = ast.literal_eval('[' + ' '.join(components[4:]) + ']')  # Join remaining parts and parse as list

                    freq = int(freq)
                    print(doc_id, term, freq, weight, pos)

                    # Apply stemming if needed
                    if stemmer_value == "Porter":
                        porter = nltk.PorterStemmer()
                        query_stemmed = porter.stem(query_value)
                    elif stemmer_value == "Lancaster":
                        lancaster = nltk.LancasterStemmer()
                        query_stemmed = lancaster.stem(query_value)
                    else:
                        query_stemmed = query_value  # No stemming

                    if index_selection == 'Docs per TERM':
                        if term == query_stemmed:
                            final_list.append((len(final_list) + 1, doc_id, term, freq, weight, pos))
                    elif int(doc_id) == int(query_value):
                        voc += 1
                        size += freq
                        final_list.append((len(final_list) + 1, doc_id, term, freq, weight, pos))
    elif matching_selection=='logique':
        if(token_value=="RegEx"):
         ExpReg = nltk.RegexpTokenizer(r"(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,\-]\d+)?%?|\w+(?:[\-/]\w+)*")
         words = ExpReg.tokenize(query_value)
        else :
         words = query_value.split()
        f=evaluate_relevance(token_value,words,stemmer_value )
        final_list = sorted(f, key=lambda x: x[1], reverse=True)

        
    else:
        # Open file for tokenized query processing
      with open(f'description{token_value}{stemmer_value}.txt', 'r') as f:
        if(token_value=="RegEx"):
         ExpReg = nltk.RegexpTokenizer(r"(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,\-]\d+)?%?|\w+(?:[\-/]\w+)*")
         words = ExpReg.tokenize(query_value)
        else :
         words = query_value.split()

       
        # w=np.array((len(words),6))
        # t = np.array((len(words),6))

        w = np.zeros((len(words),6)) 
        t = np.zeros(( len(words),6)) 
        b=np.zeros(( len(words),6))
        fr = np.zeros(( len(words),6))
        sizeW=  np.zeros((6)) 


        i = 0
        
        stopwords = nltk.corpus.stopwords.words('english')

    # Initialize the Porter Stemmer
        # if stemmer_value == "Porter":
        #         porter = nltk.PorterStemmer()
        #         words =  set(porter.stem(word) for word in words if word.lower() not in stopwords)

        # elif stemmer_value == "Lancaster":
        #         lancaster = nltk.LancasterStemmer()
        #         words =  set(lancaster.stem(word) for word in words if word.lower() not in stopwords)

        if stemmer_value == "Porter":
          porter = nltk.PorterStemmer()
          words = [porter.stem(word) for word in words if word.lower() not in stopwords]

        elif stemmer_value == "Lancaster":
           lancaster = nltk.LancasterStemmer()
           words = [lancaster.stem(word) for word in words if word.lower() not in stopwords]
        print(words)
                        

        # Process all words in the 'words' list
        for word in words:
            # Apply stemming to query terms if needed
        
            all=[]

            sizeW=  np.zeros((6)) 

            print(word)
            # with open(f'description{token_value}{stemmer_value}.txt', 'r') as g:
            for line in f:
                components = line.strip().split()
                doc_id = components[0]
                term = components[1]
                freq = components[2]
                weight = components[3]
                sizeW[ int(doc_id) - 1]+=int(freq)
                all.append((float(weight),int(doc_id)-1 ))
                if term == word:
                  print(term , word) 
                  fr[i, int(doc_id) - 1]=int(freq)
                  print(weight)
                  print(i, int(doc_id)-1,float(weight))
                  w[i, int(doc_id) - 1] = float(weight)
                  t[i, int(doc_id) - 1] = 1
                  
                # else:
                #   w[i, int(doc_id) - 1] = 0
                #   t[i, int(doc_id) - 1] = 0
                    # Exit the loop once a match is found
            i += 1  
            print(i)
            print(t)
            print(w)
            f.seek(0)
            # Increment i after pr
        print(w)
        print(t)
        # print(all)
        if matching_selection == "Vector Space Model":
         rsv = []  # Initialize list to store (doc_id, relevance) tuples
         for j in range(6):
            # relevance_score = np.sum( t[:, j]*w[:, j] )
            relevance_score = np.sum( w[:, j] )
        
            rsv.append((j + 1, relevance_score))  # Store (doc_id, relevance score)
         rsv_sorted = sorted(rsv, key=lambda x: x[1], reverse=True)

         print("RSV for each document:", rsv)
         final_list = rsv_sorted 

        elif matching_selection == "Cosine Measure":
         rsv = []  # Initialize list to store (doc_id, relevance) tuples
         for j in range(6):
            filtered_values=[]

            print(j)
            dot_product = np.sum(  t[:, j]*w[:, j] )
            ts=np.sum((t[:, j]**2))
            # ts=len(t[:, j])
            print("ts",ts)

            query_norm = math.sqrt(ts)
            print(query_norm )
            
            
            # ws=np.sum((w[:, j]**2))
            for x, y in all:
                if( y == j):
                    filtered_values.append(x**2)  
            
            # filtered_values = [x for x, y in all if y == j]
            print(len(filtered_values))
            ws  = sum(filtered_values)

            print("ws",ws)

            doc_norm = math.sqrt(ws)
            print(doc_norm )
            relevance_score = (dot_product / (query_norm * doc_norm)) if query_norm != 0 and doc_norm != 0 else 0
            rsv.append((j + 1, relevance_score))  # Store (doc_id, relevance score)
        
         print("Normalized RSV for each document:", rsv)
         rsv_sorted = sorted(rsv, key=lambda x: x[1], reverse=True)

         final_list =  rsv_sorted 

        elif matching_selection == "Jaccard Measure":
         rsv = []  # Initialize list to store (doc_id, relevance) tuples
         for j in range(6):
            print(j)
            filtered_values=[]
            sum_vw = np.sum(w[:, j] * t[:, j])
            for x, y in all:
                if( y == j):
                    filtered_values.append(x**2)  
            print(len(filtered_values))
            
            sum_v2 =np.sum(t[:, j]**2)
            # np.sum((filtered_values))
            
            sum_w2 =np.sum(filtered_values)
            # np.sum((w[:, j]**2))
            relevance_score = sum_vw / (sum_v2 +sum_w2- sum_vw) if (sum_v2 + sum_w2 - sum_vw) != 0 else 0
            rsv.append((j + 1, relevance_score))  # Store (doc_id, relevance score)
        
         print("Jaccard RSV for each document:", rsv)
         rsv_sorted = sorted(rsv, key=lambda x: x[1], reverse=True)

         final_list =  rsv_sorted 
        elif matching_selection == "proba":
          k = 1.5  # Hyperparameter for term saturation
          b = 0.75  # Hyperparameter for document length normalization
          N = 6  # Total number of documents
          rsv = []  # Initialize list to store (doc_id, relevance) tuples
          print(fr)
          avdl = np.sum(sizeW) / N  # Average document length
          print(avdl)
          for j in range(6):
           doc_score = 0  # Initialize the score for document j
           dl = sizeW[j]   
           
           for i in range(len(t[:, j])):
             freq=fr[i,j]
             if freq > 0:
                ni = np.sum(t[i, :])  # Number of documents containing term i
                idf = np.log10((N - ni + 0.5) / (ni + 0.5))  # Smooth IDF
                # t=k * ((1 - b) + (b * (dl / avdl)
                print(f"Term frequency (freq): {freq}, IDF: {idf}")

            # BM25 term score
                term_score = (
                ((freq ) /
                 (freq + k * ((1 - b) + (b * (dl / avdl))))) * idf
                )
                doc_score += term_score
        
           rsv.append((j + 1, doc_score))  # Store document ID and its score
    
          print("BM25 RSV for each document:", rsv)
          rsv_sorted = sorted(rsv, key=lambda x: x[1], reverse=True)
          final_list=rsv_sorted 
    
        
    return final_list, voc, size,matching_selection



# Streamlit app layout
st.set_page_config(page_title="Search Interface", layout="wide")
st.title("Search Interface")

# Add custom CSS for dark background
st.markdown(
    """
    <style>
    .reportview-container {
        background: #2E2E2E;
        color: white;
    }
    .stDataFrame {
        background-color: #3E3E3E;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Query Input Section
st.subheader("Query")
query_value = st.text_input("", placeholder="Type your query here...")
# Processing Section
col1, col2, col3 ,col4,col5= st.columns([3, 3, 3,3,3])
with col1:
    st.subheader("Token Selection")
    token_value = st.selectbox("", ["split", "RegEx"], index=0)

with col2:
    st.subheader("Stemmer Selection")
    stemmer_value = st.selectbox("", ["Porter", "Lancaster", "without"], index=0)

with col3:
    st.subheader("Index Selection")
    index_selection = st.radio("", ["Docs per TERM", "Terms per DOC"], index=0)

with col4: 
 st.subheader("Matching")
 matching_selection = st.selectbox("", ["without","proba","Vector Space Model", "Cosine Measure", "Jaccard Measure","logique"], index=0)
with col5:
#   st.subheader("")
  K=  st.text_input("k:", "")

  B =st.text_input("b:", "")




# Search Button
search_button = st.button("Search")

# Perform the search action when the button is pressed
if search_button:
    # Perform the search action
    results, voc, size ,m = search_action(query_value, token_value, stemmer_value, index_selection, matching_selection,K,B)

    # Display the results
    if results:
        st.subheader("Results")
        col1, col2, col3 = st.columns([3, 3, 4])
        with col1:
            st.write(f"Number of results: {len(results)}")
        with col2:
            if size != 0:
                st.write(f"Doc vocabulary: {voc}")
        with col3:
            if size != 0:
                st.write(f"Doc size: {size}")
        

        # Display results in a larger DataFrame
        print(results)

        if m == "without":
            results_df = pd.DataFrame(results, columns=["N°", "N°doc", "Term", "Freq", "Weight", "Positions"])
            st.write(results_df)
        else:
            st.write(f"Results for matching method: {matching_selection}")
            results_df = pd.DataFrame(results, columns=["N°doc", "relvent"])
            
            
            st.write(results_df )
