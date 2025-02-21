import numpy as np
import spacy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import List

class VectorBuilder:

    def __init__(self, categories: List):
        self.categories = categories
        # Ucitavamo Engleski tokenizer, parser i tagger
        self.nlp = spacy.load("en_core_web_sm")

    def generator(self):
        raw_data_list = ['dog', 'cat' ,'hamster', 'lion', 'tiger', 'elephant', 'cheetah', 'monkey', 'gorilla', 'antelope', 'rabbit', 'mouse', 'rat', 'zoo', 'home', 'pet', 'fluffy', 'wild', 'domesticated']

        # self.categories #['dog', 'cat' ,'hamster', 'lion', 'tiger', 'elephant', 'cheetah', 'monkey', 'gorilla', 'antelope', 'rabbit', 'mouse', 'rat', 'zoo', 'home', 'pet', 'fluffy', 'wild', 'domesticated']

        converted_data = " ".join(str(x) for x in raw_data_list)

        data_tokens = self.nlp(converted_data)
        _vectors = np.vstack([word.vector for word in data_tokens if word.has_vector])
        # print(_vectors)
        pca = PCA(n_components=1) # u dva reda/dve koordinate
        _vecs_transformed = pca.fit_transform(_vectors)
        # print(_vecs_transformed)
        flat_list = [float(x) for xs in _vecs_transformed for x in xs]
        # print(flat_list)


        x = (lambda counter: sum(0 for i in raw_data_list if i == " ") + counter)
        prostor = np.array(range(0, x))
        vrednosti = np.array(flat_list)

        plt.scatter(prostor, vrednosti)

        plt.show()

    #Cosine Similarity in a Vector Space Model

if __name__=='__main__':
    v = VectorBuilder([3,5])
    v.generator()


#|a| and |b| are the magnitudes (lengths) of the vectors, and θ is the angle between them

#dot_product(a,b) = |a| × |b| × cos(θ)

# cosine_similarity(a,b) = dot_product(a,b) / (|a| × |b|)


        # x = (lambda counter: sum(0 for i in raw_data_list if i == " ") + counter)
        # prostor = np.array(range(0, x))
        # vrednosti = np.array(flat_list)
