import streamlit as st
import pandas as pd
import nltk
import numpy as np
import math
import re
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
import ast

import boolean

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
selected=[(1, [13, 14, 15, 72, 79, 138, 142, 164, 165, 166, 167, 168, 169, 170, 171, 172, 180, 181, 182, 183, 184, 185, 186, 211, 212, 499, 500, 501, 502, 503, 504, 506, 507, 508, 510, 511, 513]), (2, [80, 90, 162, 187, 236, 237, 258, 289, 290, 292, 293, 294, 296, 300, 301, 303]), (3, [59, 62, 67, 69, 70, 71, 73, 78, 81, 160, 163, 230, 231, 232, 233, 234, 276, 277, 279, 282, 283, 287]), (4, [93, 94, 96, 141, 173, 174, 175, 176, 177, 178, 207, 208, 209, 210, 259, 396, 397, 399, 400, 404, 405, 406, 408]), (5, [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 158, 159, 188, 304, 305, 306, 307, 325, 326, 327, 329, 330, 331, 332, 333]), (6, [112, 115, 116, 118, 122, 238, 239, 242, 260, 309, 320, 321, 323]), (7, [92, 121, 189, 247, 261, 382, 385, 386, 387, 388, 389, 390, 391, 392, 393]), (8, [52, 60, 61, 123, 190, 251, 262, 263, 264, 265, 266]), (9, [30, 31, 53, 56, 57, 64, 83, 84, 89, 124, 125, 126, 192, 252, 253, 267, 268, 269, 270, 271, 272, 273, 409, 412, 415, 420, 421, 422]), (10, [54, 55, 58, 152, 153, 154, 155, 254, 255, 256, 257, 529, 531, 532, 533, 534, 535, 537, 538, 539, 540, 541, 542, 543]), (11, [32, 63, 66, 148, 150, 225, 226, 228, 229, 440, 441, 444, 445, 446, 447, 448, 451, 452]), (12, [16, 17, 19, 20, 193, 364, 365, 366, 367]), (13, [21, 22, 143, 144, 145, 146, 194, 195, 196, 197, 198, 199, 470, 471, 474, 475, 477, 478, 479, 481, 483]), (14, [23, 24, 25, 26, 28, 29, 454, 455, 456, 457, 459, 461, 463, 466, 467, 468]), (15, [33, 34, 101, 102, 104, 105, 107, 109, 110, 140, 215, 216, 218, 219, 220, 222, 349, 350, 351, 352, 353, 355, 356, 357, 358, 359, 361, 362, 363]), (16, [35, 36, 98, 99, 202, 205, 484, 487, 488, 490, 492, 493, 495]), (17, [37, 38, 39, 41, 42, 127, 129, 130, 131, 132, 133, 334, 335, 337, 338, 339, 340, 341, 342, 346, 348]), (18, [43, 514, 515, 516, 517, 518, 519, 521, 522, 523, 524, 525, 526, 527, 528]), (19, [544, 545, 549, 550, 555, 560, 562, 563, 564, 565, 566, 844, 845, 846, 847, 854, 856, 857, 858, 859, 860, 861, 862, 864, 865, 866, 867]), (20, [567, 570, 571, 573, 574, 575, 576, 577, 578, 580, 581, 584, 585, 588, 589, 590, 593, 594, 595, 596, 597, 598, 599, 601, 602, 848, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 883]), (21, [604, 605, 608, 610, 612, 613, 615, 616, 618, 619, 620, 622, 626, 630, 631, 884, 885, 886, 888, 890, 891, 892, 893, 894, 895, 896, 898]), (22, [633, 635, 636, 638, 640, 641, 643, 644, 645, 647, 648, 651, 652, 654, 655, 656, 657, 658, 659, 660, 899, 900, 901, 904, 907]), (23, [797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 849, 914, 915, 916, 917, 918, 919, 921, 922, 923, 924, 927, 928]), (24, [663, 666, 667, 668, 670, 674, 682, 684, 686, 850, 851, 852, 929, 930, 932, 935, 936, 938, 940, 941, 942, 943]), (25, [687, 688, 689, 690, 691, 692, 693, 694, 695, 697, 698, 699, 944, 945, 947, 948, 949, 951, 952, 953, 954, 955, 956, 958]), (26, [708, 712, 713, 714, 715, 716, 717, 719, 721, 722, 723, 724, 725, 726, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 972, 973]), (27, [727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 769, 975, 977, 980, 984]), (28, [770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 795, 796, 989, 990, 991, 992, 993, 994, 995, 996, 997, 999, 1000, 1001, 1002, 1003]), (29, [740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 754, 757, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 853, 1004, 1006, 1007, 1008, 1009, 1010, 1012, 1013, 1015, 1016, 1017]), (30, [823, 825, 827, 831, 843, 1019, 1020, 1021, 1022, 1024, 1026, 1027, 1032, 1033])]

def interpoler_precisions(precision_rappel, rappel_interpole):
    precision_interpolee = []
    for rj in rappel_interpole:
        precisions_filtrees = [precision for precision, rappel in precision_rappel if rappel >= rj]
        precision_interpolee.append(max(precisions_filtrees) if precisions_filtrees else 0)
    return precision_interpolee

def precision(sect,filtred):
    s = 0  # Initialisation du compteur
    top5_ids = [id for id, _ in filtred]  # Prendre les 5 premiers IDs filtrés
    
    for i in sect:
        if i in top5_ids:  # Vérifier si l'ID pertinent est dans les 5 premiers
            s += 1
    if len(filtred) == 0:
        return 0  # Eviter la division par zéro
    return s / len(filtred)
def precision5(sect,filtred):
    s = 0  # Initialisation du compteur
    top5_ids = [id for id, _ in filtred[:5]]  # Prendre les 5 premiers IDs filtrés
    
    for i in sect:
        if i in top5_ids:  # Vérifier si l'ID pertinent est dans les 5 premiers
            s += 1
    
    return s / 5 
  # Eviter la division par zéro
def precision6(sect, filtred):
    s = 0  # Initialisation du compteur
    top5_ids = [id for id, _ in filtred[:10]]  # Prendre les 5 premiers IDs filtrés
    
    for i in sect:
        if i in top5_ids:  # Vérifier si l'ID pertinent est dans les 5 premiers
            s += 1
    
    return s / 10
 

# Fonction pour calculer le rappel
def recall(sect, filtred):
    s = 0  # Initialisation du compteur
    top5_ids = [id for id, _ in filtred]  # Prendre les 5 premiers IDs filtrés
    
    for i in sect:
        if i in top5_ids:  # Vérifier si l'ID pertinent est dans les 5 premiers
            s += 1
    if len(sect) == 0:
        return 0  # Eviter la division par zéro
    return s / len(sect)
# Fonction pour calculer le F-score
def f_score(precision_value, recall_value):
    if precision_value + recall_value == 0:
        return 0  # Eviter la division par zéro
    return 2 * (precision_value * recall_value) / (precision_value + recall_value)

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
    final = []
    s=[]
    for doc_id in range(6):
        # Convert column vector into a boolean list or keep operators
        bool_vector = [
            'True' if x == 1 else 'False' if x == 0 else x
            for x in t[:, doc_id]
        ]
       
          
             
             
        print(bool_vector)
        b=" ".join(bool_vector)
        print(b)

      
         
        # result = expr.evaluate()
        result= eval(b)
        print (result)

        final.append((doc_id + 1, str(result)))
    
    return final
  

def search_action(query_value, token_value, stemmer_value, index_selection, matching_selection,k,b,eva,q):
    final_list = []
    size = 0
    voc = 0
    rel=[]
    fig=None
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
        if stemmer_value == "Porter":
                porter = nltk.PorterStemmer()
                words =  set(porter.stem(word) for word in words if word.lower() not in stopwords)

        elif stemmer_value == "Lancaster":
                lancaster = nltk.LancasterStemmer()
                words =  set(lancaster.stem(word) for word in words if word.lower() not in stopwords)

            
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
            # np.sum((w[:, j]**2))
            # sum(filtered_values)

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
            filtered_values=[]
            sum_vw = np.sum(w[:, j] * t[:, j])
            for x, y in all:
                if( y == j):
                    filtered_values.append(x**2)  
            
            sum_v2 =np.sum(t[:, j]**2)
            # np.sum((filtered_values))
            
            sum_w2 = np.sum(filtered_values)
            # np.sum((w[:, j]**2))
            # np.sum(filtered_values)
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
        if eva=='OUI': 
             filtred = [(id, val) for id, val in final_list if val != 0]
             
             _,sect=selected[int(q)-1]
             print(sect)
             sect = [1,2, 4]  # Liste d'identifiants pertinents
    
             # Calcul des métriques
             p = precision(sect,filtred)
             p5 = precision5(sect,filtred)  # Hypothèse de precision@5
             p6 = precision6(sect,filtred)  # Hypothèse de precision@6
             r = recall(sect,filtred)
             f = f_score(p, r)
             rel = [(p, p5, p6, r, f)]
             i = 0
             sel = 0
             total = []
             for id, val in filtred:
              i += 1
              if id in sect:
                sel += 1
              prs = sel / i  # Précision
              rec = sel / len(sect)  # Rappel
              total.append((prs, rec))
             rappel_interpole = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
             precision_interpolee = interpoler_precisions(total, rappel_interpole)
             fig, ax = plt.subplots()
             ax.plot(rappel_interpole, precision_interpolee, marker="o", linestyle="-", color="blue")
             ax.set_title("Précision Interpolée en Fonction du Rappel")
             ax.set_xlabel("Rappel (rj)")
             ax.set_ylabel("Précision (Pr)")
             ax.grid()
        
    return final_list, voc, size,matching_selection,rel,fig



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

col22,col23=st.columns([2,2])
with col22:
 st.subheader("EVAL")
 eva = st.radio("", ["NON", "OUI"], index=0)
with col23:
    q =st.text_input("query number:", "")
    



# Search Button
search_button = st.button("Search")

# Perform the search action when the button is pressed
if search_button:
    # Perform the search action
    results, voc, size ,m ,rel,fig= search_action(query_value, token_value, stemmer_value, index_selection, matching_selection,K,B,eva,q)

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
        if len(rel)!=0:
          p, p5, p6, r, f= rel[0]
          st.write(f"Précision : {p}, Précision@5 : {p5}, Précision@10 : {p6}, Rappel : {r}, F-Score : {f}")

        # Display results in a larger DataFrame
        print(results)
        print(rel)
        col11, col12=st.columns([4,4])
        with col11:
         if m == "without":
            results_df = pd.DataFrame(results, columns=["N°", "N°doc", "Term", "Freq", "Weight", "Positions"])
            st.write(results_df)
         else:
            st.write(f"Results for matching method: {matching_selection}")
            results_df = pd.DataFrame(results, columns=["N°doc", "relvent"])
            
            
            st.write(results_df )
        with col12:
            if fig !=None:
                 st.pyplot(fig)
