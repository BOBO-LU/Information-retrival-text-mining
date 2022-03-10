import pa2
import numpy as np
from typing import TypeVar
from tqdm import tqdm

if __name__ == "__main__":

    # variables
    DATA_PATH_BASE = "./data/"
    OUTPUT_PATH_BASE = "./output/"
    T = TypeVar('T')

    TFIDF = pa2.TF_IDF(display=False, stem_min=0)

    TFIDF.get_file_from_folder(DATA_PATH_BASE)  # read all files from folder
    TFIDF.convert_all_document()  # convert every file to DOCUMENT
    TFIDF.calc_idf()  # calculate idf
    TFIDF.calc_docuement_tf_idf()  # calculate tfidf of each DOCUMENT
    _ = TFIDF.calc_all_cosine_similarity()



    # initialize
    I = np.ones(TFIDF.num_of_documents)
    A = np.empty((0,2))
    C = np.array(TFIDF.cosine_matrix)
    clus_i, clus_j = -1, -1
    result = []
    # set diagnoal to -1 in order to prevent select it
    np.fill_diagonal(C, -1)


    for k in tqdm(range(TFIDF.num_of_documents - 1)):


        # find two most similar clusters
        sort_cord = np.unravel_index(np.argsort(C, axis=None, kind="quicksort")[::-1],C.shape)
        for cord in zip(sort_cord[0], sort_cord[1]):
            _i, _j = cord[0], cord[1]
            if I[_i] == 1 and I[_j] == 1:
                clus_i, clus_j = min(_i, _j), max(_i, _j) 
                break

        # debug
        if clus_i == -1 or clus_j == -1:
            print("still -1")

        # record the pair and change I matrix
        A = np.append(A, [[ clus_i, clus_j ]], axis=0)
        I[max(clus_i, clus_j)] = 0

        # update the matrix
        for m in range(TFIDF.num_of_documents):
            new_sim = min(C[m][clus_i], C[m][clus_j])
            C[m][clus_i] = new_sim
            C[clus_i][m] = new_sim

        # save result
        if I.sum() in [8, 13, 20]:
            result.append({"A":A.copy(), "I":I.copy(), "C":C.copy()})



    for res in result:
        k = res['I'].sum()
        clusters = [ [x] for x in np.where( res['I'] == 1 )[0] ]
        for a in res['A'][::-1]: # iter every append action
            clus_i, clus_j = a
            for c in range(len(clusters)): # c is the index, iter every clusters
                if clus_i in clusters[c]: 
                    clusters[c].append(clus_j)
        with open(str(int(k)) +  ".txt", 'w+') as out_file:
            for clus in clusters:
                for num in sorted(clus):
                    out_file.write(str(int(num)+1)+"\n")
                out_file.write("\n")

    print("finish")
