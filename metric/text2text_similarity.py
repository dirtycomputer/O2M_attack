# text_similarity.py
from FlagEmbedding import BGEM3FlagModel
import os
import pandas as pd

class TextSimilarityCalculator:
    def __init__(self, model_path='BAAI/bge-m3', use_fp16=True):
        self.model = BGEM3FlagModel(model_path, use_fp16=use_fp16)

    def read_texts_from_folder(self, folder):
        texts = {}
        for filename in os.listdir(folder):
            if filename.endswith('.txt'):
                filepath = os.path.join(folder, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    texts[filename] = file.read()
        return texts

    def compute_similarities_folder(self, normal_text_folder, harmful_text_folder):
        normal_texts = self.read_texts_from_folder(normal_text_folder)
        harmful_texts = self.read_texts_from_folder(harmful_text_folder)

        sentence_pairs = []
        file_pairs = []
        for filename, normal_text in normal_texts.items():
            if filename in harmful_texts:
                harmful_text = harmful_texts[filename]
                sentence_pairs.append([normal_text, harmful_text])
                file_pairs.append(filename)

        if sentence_pairs:
            similarity_scores = self.model.compute_score(sentence_pairs, max_passage_length=128, weights_for_different_modes=[0.4, 0.2, 0.4])
            df_scores = pd.DataFrame(similarity_scores)
            df_scores['File Pair'] = file_pairs
            df_scores = df_scores[['File Pair', 'colbert', 'sparse', 'dense', 'sparse+dense', 'colbert+sparse+dense']]
            return df_scores
        else:
            return "No matching files found."
    
    def compute_similarity_text(self, text1, text2):
        similarity_score = self.model.compute_score([[text1, text2]], max_passage_length=128, weights_for_different_modes=[0.4, 0.2, 0.4])
        df_scores = pd.DataFrame(similarity_score)
        df_scores = df_scores[[ 'colbert', 'sparse', 'dense', 'sparse+dense', 'colbert+sparse+dense']]
        return df_scores


