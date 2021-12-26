import numpy as np
import os
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter, OrderedDict
from typing import TypeVar, Generic, Union, List
from math import log

# variables
DATA_PATH_BASE = "./data/"
OUTPUT_PATH_BASE = "./output/"
T = TypeVar('T')


class TOKENIZE:

    def __init__(self, txt=""):
        '''Initialize all the objects on class instantiation.'''
        self.txt = txt
        self.tokens = []

    def tokenization(self) -> None:
        '''Tokenize tokens in PA1.'''
        valid = ""

        # Save alphabets and space into a new string called valid
        for character in self.txt:
            if character.isalpha() or character.isspace():
                valid += (character)

        # Split the string "valid" by space into list
        self.tokens = valid.split()

    def lowercasing(self) -> None:
        '''Lowercase tokens in PA1.'''

        # For every token inside "tokens", lowercase it
        for idx, token in enumerate(self.tokens):
            self.tokens[idx] = token.lower()

    def steamming_with_porter(self, m=0) -> None:
        '''
        Steamming tokens in PA1 with porter algorithm.
        if words char was not larger then m, then don't stem
        '''

        # Initialzie porter algorithm as "ps"
        ps = PorterStemmer()

        # For every token inside "tokens", use "ps" to stem it
        for idx, token in enumerate(self.tokens):
            s = ps.stem(token)

            # if m is zero, or length after stem is larger then m
            if not m or len(s) > m:
                self.tokens[idx] = s

    def stopword_removal(self) -> None:
        '''Remove all stopwords in tokens in PA1.'''
        clean_tokens = []

        # Declare the stopwords
        STOP_WORDS = ['were', 'shan', 'them', 'i', 'just', 'him', 'below', 'both', "you'd",
                      'don', 'wouldn', "wouldn't", 'then', 's', 'will', 'how', 'wasn', 'am',
                      'should', 'from', 'hasn', 'each', 'any', 'yours', 'who', 'such', 'can',
                      'once', 'on', 'all', 'haven', 'didn', 'again', 'ain', 'doesn', 'same',
                      't', 'and', 'hers', "weren't", 'until', 'has', 'themselves', 'in', 'she',
                      "you've", "couldn't", 'a', 'about', 'been', 'because', 'herself', 'ourselves',
                      'with', 'isn', "didn't", 'mustn', 'needn', 'or', "you're", 'but', 'you',
                      "mightn't", 'of', 'under', 'where', "that'll", 'which', 'does', "hasn't",
                      'have', 'mightn', "hadn't", 'y', 'what', 'won', 'he', 'nor', 'between',
                      'couldn', 'an', 'whom', 'than', 'no', 'd', 'yourself', 'only', 'the', 'this',
                      'after', 'her', "shan't", 'itself', 'being', 'do', 'against', 'into',
                      'me', 'over', "haven't", "don't", 'your', 'now', 'we', 'aren', 'some',
                      'why', 'very', 'shouldn', 'ours', 'doing', 'ma', 'off', 'there', 'himself',
                      "isn't", 'at', 'during', 'had', 'too', 'my', 'before', 'it', 'while', 'most',
                      "doesn't", 'few', 'be', 'hadn', 'those', 'theirs', 'its', 'here', 'll',
                      "you'll", 'myself', 'further', 're', 'their', 'they', "should've", "it's",
                      'by', 'his', 'are', 'yourselves', 'through', 've', 'above', 'o', 'when',
                      'so', "shouldn't", 'out', 'm', "aren't", 'these', 'not', 'weren', 'did',
                      'own', 'if', 'is', 'having', 'as', "mustn't", "won't", 'up', 'our', "she's",
                      'was', 'down', "needn't", 'more', 'that', "wasn't", 'to', 'other', 'for']

        # Append token from "tokens" which does not exist in "STOP_WORDS" into a new list
        for token in self.tokens:
            if token not in STOP_WORDS:
                clean_tokens.append(token)

        # Replace the tokens in PA1 with the one without any stopwords
        self.tokens = clean_tokens

    def save_result(self) -> None:
        '''Save the tokens in PA1 into text file named result.txt.'''

        # Open a textfile and write the tokens into it
        with open("result.txt", "w") as f:
            data = " ".join(self.tokens)
            f.write(data)

    def print_tokens(self, display=False) -> None:
        '''Print tokens in PA1.'''

        # Print the object PA1's tokens
        if display:
            print(self.tokens)

    def tokenize(self, corpus, display, stem_min=0) -> list:
        '''tokenize the corpus'''

        self.txt = corpus
        self.tokenization()  # Tokenization.
        self.lowercasing()  # Lowercasing everything.
        self.stopword_removal()  # Stopword removal.
        # Stemming using Porter’s algorithm.
        self.steamming_with_porter(stem_min)
        self.print_tokens(display)  # Print tokens.
        return self.tokens


class DOCUMENT(Generic[T]):
    def __init__(self, filepath, name, id, display=False, stem_min=0):
        self.TK = TOKENIZE()

        self.filepath = filepath
        self.name = name
        self.id = id
        self.display = display
        self.corpus = self.read_file()  # original words in the docuemnt, string
        self.tokens = self.get_tokens(stem_min)  # tokens of the document, list
        self.tf = self.calc_frequency()  # frequency of each tokens, dict
        self.all_tf_dict = {}  # term frequency in all words, dict
        self.all_tf_list = []  # term frequency list in all words, list, sort in alphabetic
        self.tfidf_unit_vector = []  # the tfidf unit vector of the document, np.ndarray

    def read_file(self) -> str:
        '''read file and turn into corpus(a string)'''

        corpus = ""
        with open(self.filepath) as in_file:
            for line in in_file:
                corpus += line
                corpus += ","
        return corpus

    def get_tokens(self, stem_min=0) -> list:
        '''tokenize the corpus'''

        return self.TK.tokenize(self.corpus, self.display, stem_min)

    def calc_frequency(self) -> dict:
        '''calcualte term frequency in the document'''

        freq = defaultdict(int)
        for token in self.tokens:
            freq[token] += 1
        return freq

    def create_all_words_term_freq(self, all_words_tf_list) -> int:
        '''create term frequency on all words in document'''

        tfd = defaultdict(int)
        for term in all_words_tf_list:
            if term in self.tokens:
                tfd[term] = self.tf[term]
            else:
                tfd[term] = 0

        self.all_tf_dict = tfd

        order_list = self.calc_ordered_freq_list(tfd)
        self.all_tf_list = order_list
        return order_list

    def calc_ordered_freq_list(self, freq_dict) -> list:
        '''create ordered term freq list by dictionary'''

        ordered_list = []
        x = list(sorted(freq_dict.items(),
                 key=lambda item: item[0], reverse=False))  # sort by key

        for i in range(len(x)):
            ordered_list.append(x[i][1])
        return ordered_list

    def calc_tf_idf_unit_vector(self, idf) -> list:
        '''calculate tfidf unit vector base on all words' idf'''

        v = np.array(self.all_tf_list)  # tf
        u = np.array(idf)  # idf
        s = np.multiply(v, u)  # tf*idf
        norm = np.linalg.norm(s)
        unit_vector = s / norm
        self.tfidf_unit_vector = unit_vector
        return unit_vector


class TF_IDF():
    def __init__(self, /, display=False, stem_min=0) -> None:

        self.display = display  # print result while running
        self.stem_min = stem_min
        self.all_files = []  # list all document file details in the collection
        self.all_documents = []  # all document object in the collection
        self.all_words = []  # the set of words in the collection
        self.df = []  # document frequency of all_words
        self.idf = []  # inverse document frequency of all_words
        self.words_freq_dict = {}  # the key is the word, value is the freq
        self.num_of_documents = 0  # how many documents/files in the collection
        # ordered freq list, [(id, term, freq), ...]
        self.ordered_freq_list = []
        self.cosine_matrix = []  # cosine matrix

    def get_file_from_folder(self, path) -> list:
        '''read all files in the folder's path and save it as a list'''

        listdir = os.listdir(path)
        files = [""] * len(listdir)
        for idx, file_name in enumerate(listdir):
            if file_name[-4:] != ".txt":
                continue
            file_path = os.path.join(path, file_name)
            # the file starts from 1, not zero
            file_num = int(file_name.split(".")[0])
            files[idx] = {"file_path": file_path,
                          "file_name": file_name, "file_num": file_num}

        self.all_files = files
        self.num_of_documents = len(files)
        return files

    def convert_single_document(self, n) -> T:  # n is document id
        '''convert one single file to a DOCUMENT instance'''
        f = 0
        for temp in self.all_files:
            if temp.file_num == n:
                f = temp
                break
        if f == 0:
            print(f"document with ID: {n} doesn't exist")
            return ""
        return DOCUMENT(f['file_path'], f['file_name'], f['file_num'], self.display, self.stem_min)

    def convert_all_document(self) -> list:
        '''convert all files into DOCUMENT instances'''

        documents = []
        for f in self.all_files:
            documents.append(DOCUMENT(
                f['file_path'], f['file_name'], f['file_num'], self.display, self.stem_min))
        self.all_documents = documents
        return self.all_documents

    def calc_idf(self) -> Union[list, list, Counter, list]:
        '''calculate the idf of the dictionary'''

        # add unique tokes into word_list to count document frequency
        word_list = []
        for d in self.all_documents:
            word_list += set(d.tokens)

        # count the term's document frequency and sort the dict
        c = Counter(word_list)
        sorted_c = sorted(c.items(), key=lambda item: item[0], reverse=False)
        counter = OrderedDict(sorted_c)
        self.words_freq_dict = counter

        # create unique word list contains any word exist in the collection
        unique_word_list = list(counter.keys())
        self.all_words = unique_word_list

        # create document frequency list
        df = list(counter.values())
        self.df = df

        # calc idf
        idf = [0] * len(df)
        for idx, doc_freq in enumerate(df):
            idf[idx] = log(self.num_of_documents / doc_freq, 10)
        self.idf = idf

        return unique_word_list, df, counter, idf

    def calc_docuement_tf_idf(self) -> None:
        '''calculate tfidf of each DOCUMENT'''

        for d in self.all_documents:
            d.create_all_words_term_freq(self.all_words)
            d.calc_tf_idf_unit_vector(self.idf)

    def calc_all_cosine_similarity(self) -> List[List[float]]:
        '''calculate the cosine similarity for all pairs of DOCUMENT and save it in a adjacent matrix'''

        n = self.num_of_documents
        cosine_matrix = [[0] * n for i in range(n)]
        for i in range(n):
            for j in range(n):
                #
                if i == j:
                    cosine_matrix[i][j] = 1
                    continue
                if i > j:
                    cosine_matrix[i][j] = cosine_matrix[j][i]
                    continue

                # mutiply two unit vectors
                x = self.all_documents[i].tfidf_unit_vector
                y = self.all_documents[j].tfidf_unit_vector

                cosine_matrix[i][j] = np.inner(x, y)
        self.cosine_matrix = cosine_matrix
        return cosine_matrix

    def get_ordered_freq_list(self) -> list:
        '''convert the dictionary into ordered freqency list'''

        ordered_list = []
        x = list(sorted(self.words_freq_dict.items(),
                 key=lambda item: item[0], reverse=False))

        for i in range(len(x)):
            ordered_list.append(x[i][1])
        self.ordered_freq_list = ordered_list
        return ordered_list

    def save_dictionary(self) -> None:
        '''save the dictionary to dictionary.txt'''

        with open('dictionary.txt', 'w') as f:
            max_len = max(len(l) for l in self.words_freq_dict.keys())
            f.write(f"t_index\t\t{'term'.ljust(max_len+1)}df\n")
            for idx, (word, freq) in enumerate(self.words_freq_dict.items()):
                f.write(f"{idx}\t\t\t{word.ljust(max_len+1)}{freq}\n")

    def save_single_tfidf(self, docid) -> None:
        '''save the tfidf unit vector of the specific DOCUMENT'''

        d = 0
        for temp in self.all_documents:
            if temp.id == docid:
                d = temp
                break
        if d == 0:
            print(f"document with ID: {docid} doesn't exist")
            return ""

        out_filename = f"{OUTPUT_PATH_BASE}{docid}.txt"
        with open(out_filename, 'w') as out_file:
            out_file.write(str(len(d.tf))+'\n')
            out_file.write("t_index tf-idf\n")
            for idx, val in enumerate(d.tfidf_unit_vector):
                if val != 0:
                    out_file.write(f"{idx} {val}\n")

    def save_all_tfidf(self) -> None:
        '''save all tfidf unit vectors'''

        for d in self.all_documents:
            out_filename = f"{OUTPUT_PATH_BASE}{d.id}.txt"
            with open(out_filename, 'w') as out_file:
                out_file.write(str(len(d.tf))+'\n')
                out_file.write("t_index tf-idf\n")
                for idx, val in enumerate(d.tfidf_unit_vector):
                    if val != 0:
                        out_file.write(f"{idx} {val}\n")


def cosine_similarity(docX_ID, docY_ID) -> int:
    '''given two docID, calculate the cosine similarity of two files'''

    x_dict, y_dict = OrderedDict(), OrderedDict()  # initialize

    # read file and turn into dict
    with open(f"{OUTPUT_PATH_BASE}{docX_ID}.txt") as fileX, \
            open(f"{OUTPUT_PATH_BASE}{docY_ID}.txt") as fileY:

        for line in fileX.readlines()[2:]:
            idx, tfidf = line.split()
            x_dict[int(idx)] = float(tfidf)
        for line in fileY.readlines()[2:]:
            idx, tfidf = line.split()
            y_dict[int(idx)] = float(tfidf)

    # get max element
    x_max, y_max = max(list(x_dict.keys())), max(list(y_dict.keys()))
    max_element = int(x_max if x_max > y_max else y_max)
    x, y = np.zeros(max_element), np.zeros(max_element)

    # convert to two list
    for idx, tfidf in x_dict.items():
        x[idx-1] = tfidf
    for idx, tfidf in y_dict.items():
        y[idx-1] = tfidf

    # calculate cosine similarity
    cos_sim = np.inner(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return cos_sim


if __name__ == "__main__":

    TFIDF = TF_IDF(display=False, stem_min=0)

    TFIDF.get_file_from_folder(DATA_PATH_BASE)  # read all files from folder
    TFIDF.convert_all_document()  # convert every file to DOCUMENT
    TFIDF.calc_idf()  # calculate idf
    TFIDF.calc_docuement_tf_idf()  # calculate tfidf of each DOCUMENT

    # pa2-1
    TFIDF.save_dictionary()  # save dictionary as dictionary.txt

    # pa2-2
    TFIDF.save_all_tfidf()  # save all tfidf unit vectors

    # pa2-3
    docX = 1
    docY = 2
    TFIDF.save_single_tfidf(docX)  # save the specific tfidf unit vector
    TFIDF.save_single_tfidf(docY)  # save the specific tfidf unit vector
    cos_sim = cosine_similarity(docX, docY)
    # print the cosine similarity of two docID
    print(
        f"The cosine similarity between doc{docX}.txt and doc{docY}.txt is {cos_sim}")
