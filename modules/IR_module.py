import json
from bm25.rank_bm25 import BM25Okapi
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from pynori.korean_analyzer import KoreanAnalyzer
from konlpy.tag import Kkma
import numpy as np
import pandas as pd

class IR_module(object):
    def __init__(self):
        model_path = './output/training_stsbenchmark_skt_kobert_model_-2022-09-29_00-18-09'
        data_path = './KorQuADDataset/KorQuAD_v1.0_dev.json'

        with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
            dataset_json = json.load(f)
            dataset = dataset_json['data']

        # self.embedder = SentenceTransformer("jhgan/ko-sbert-sts")
        self.embedder = SentenceTransformer(model_path)
        self.corpus = []
        self.question = []
        self.QA = []

        for article in dataset:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    context = paragraph['context']
                    question = qa['question']

                    self.corpus.append(context)
                    self.question.append(question)
                    self.QA.append({"질문" : question,
                                    "문단" : context})

        self.corpus = list(set(self.corpus))
        self.kkm = Kkma()
        self.nori = KoreanAnalyzer(
            decompound_mode='DISCARD',  # DISCARD or MIXED or NONE
            infl_decompound_mode='DISCARD',  # DISCARD or MIXED or NONE
            discard_punctuation=True,
            output_unknown_unigrams=False,
            pos_filter=False, stop_tags=['JKS', 'JKB', 'VV', 'EF'],
            synonym_filter=False, mode_synonym='NORM',  # NORM or EXTENSION
        )

        # tokenized_corpus = [self.kkm.morphs(doc) for doc in self.corpus]
        tokenized_corpus = [self.nori.do_analysis(doc)['termAtt'] for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def search(self, query, topk=10, mode=None):
        tokenized_query = self.nori.do_analysis(query)['termAtt']
        # tokenized_query = self.kkm.morphs(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        doc_scores = self.softmax(doc_scores)
        document_idx = np.argpartition(-doc_scores, range(topk))[0:topk]
        top_documents = [self.corpus[i] for i in document_idx]
        doc_scores = [doc_scores[i] for i in document_idx]

        if mode!='evaluate':
            print("Question :", query)
            print("Top Sentence :", top_documents)
            print("Score :", round(max(doc_scores), 3))

        return top_documents, doc_scores

    def reranking(self, query, top_documents, doc_scores, top_k, alpha = 0.9, mode=None):
        corpus_embeddings = self.embedder.encode(top_documents, convert_to_tensor=True)
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        cos_scores = cos_scores.cpu().numpy()
        cos_scores = self.softmax(cos_scores)
        result_scores = (1-alpha)*np.array(doc_scores) + alpha*cos_scores
        top_results = np.argpartition(-result_scores, range(top_k))[0:top_k]
        rerank_documents = [top_documents[i] for i in top_results]
        rerank_scores = [result_scores[i] for i in top_results]

        if mode!='evaluate':
            print("Reranking Top Sentence :", rerank_documents[0])
            print("Reranking Score :", round(max(rerank_scores), 3))

        return rerank_scores, rerank_documents

    def evaluate(self, topk=10, mode=None):

        predict_lists = []

        for qc_pair in tqdm(self.QA, desc="Evaluate"):
            query = qc_pair["질문"]
            correct_context = qc_pair["문단"]
            top_documents, doc_score = self.search(query, topk, mode=mode)
            _,top_documents = self.reranking(query, top_documents, doc_score, topk, mode=mode)

            bool_documents = [0]*topk
            for idx, document in enumerate(top_documents):
                bool_documents[idx] = int(correct_context == document)

            predict_lists.append(bool_documents)

        print('MRR Score :',self.mrr_measure(predict_lists))
        print('Top10 Precision Score :',self.top_N_precision(predict_lists,topk=topk))

    def mrr_measure(self,predict_list):
        score = 0
        for predict in predict_list:
            if 1 not in predict:
                continue
            score += 1 / (predict.index(1) + 1)
        return score / len(predict_list)

    def top_N_precision(self, predict_list, topk):
        c, m = [0] * topk, 0
        for idx, predict in enumerate(predict_list):
            if 1 in predict:
                c[predict.index(1)] += 1
            m += 1
        top_n_precision = [sum(c[:idx + 1]) / m for idx, e in enumerate(c)]

        return top_n_precision