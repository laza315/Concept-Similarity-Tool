import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import gensim.downloader as api
from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
import torch


nltk.download('punkt')  

class VectorSpaceModel:

    def __init__(self, documents_to_compare: list, query: str):

        self.documents = documents_to_compare  
        self.query = query
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    def concept_comparator(self) -> list:
        from nltk.tokenize import word_tokenize

        tokenized_documents = [' '.join(word_tokenize(doc.lower())) for doc in self.documents]
        tokenized_query = ' '.join(word_tokenize(self.query.lower()))

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(tokenized_documents)
        
        # Transformisemo query u TF-IDF vector. To je Term Frequency jedne reci(term-a) po single dokumentu
        query_vector = tfidf_vectorizer.transform([tokenized_query])

        '''cosine_similarity(a,b) = dot_product(a,b) / (|a| × |b|) je zapravo ugao θ izmedju 2 vektora
        Na zivom primeru: Recimo dva vektora u x,y prostoru
        |
        |    *A(10,2.5)
        |
        |          *B(1.5,10)
        |____ _____ _____
        dot_product formula je proizvod vektora A i B = 10*1.5+2.5*10 / sa proizvodom njenih smerova(magnituda) koren od A2 * koren B2 . 
        Drugim recima cosine_sim = |a| × |b| × cos(θ) . Rezultat daje ugao u radijanima i predstavlja rastojanje([ne]slicnost) dva vektora u vcm prostoru u odnosu na tacku 0'''
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)[0]

        
        results = []
        for i in range(len(self.documents)):
            results.append({"document": self.documents[i], "similarity_score": round(cosine_similarities[i], 2)})


        results.sort(key=lambda x: x["similarity_score"], reverse=True)  

        return results
    

    
    def concept_similarity(self) -> list:
        results = []

        for concept1, concept2 in self.documents:
            try:
                # Compute similarity using BART model
                similarity_score = self.bart_similarity(concept1, concept2)
                
                results.append({
                    "concepts": f"{concept1} i {concept2}",
                    "similarity": round(similarity_score, 4)
                })

            except Exception as e:
                print(f"Upozorenje: {e}. Preskacem par ({concept1}, {concept2})")
                results.append({"concepts": f"{concept1} i {concept2}", "similarity": "N/A"})

        return results

    def bart_similarity(self, concept1: str, concept2: str) -> float:
        inputs = self.tokenizer([concept1, concept2], return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            output = self.model(**inputs, decoder_input_ids=inputs['input_ids'])
        
        embeddings = output.last_hidden_state.mean(dim=1) 
        v_1 = embeddings[0].numpy()
        v_2 = embeddings[1].numpy()
        
        # Compute cosine similarity
        cosine_score = cosine_similarity([v_1], [v_2])[0][0]
        return cosine_score




        