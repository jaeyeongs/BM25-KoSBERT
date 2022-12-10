import json
from bm25.rank_bm25 import BM25Okapi
from tqdm import tqdm
from pynori.korean_analyzer import KoreanAnalyzer

class IR_base(object):
    def __init__(self):
        data_path = '/workspace/KorQuADDataset/KorQuAD_v1.0_dev.json'

        with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
            dataset_json = json.load(f)
            dataset = dataset_json['data']

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
        self.nori = KoreanAnalyzer(
            decompound_mode='DISCARD',  # DISCARD or MIXED or NONE
            infl_decompound_mode='DISCARD',  # DISCARD or MIXED or NONE
            discard_punctuation=True,
            output_unknown_unigrams=False,
            pos_filter=True, stop_tags=['MM', 'MAG', 'MAJ', 'IC', 'JKS', 'JKC', 'JKG',
                                        'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'EP',
                                        'EF', 'EC', 'ETN', 'ETM', 'XPN', 'XSN', 'XSV',
                                        'XSA', 'XR', 'SF', 'SE', 'SSO', 'SSC', 'SC',
                                        'SL', 'SH'],
            synonym_filter=False, mode_synonym='NORM',  # NORM or EXTENSION
        )

        tokenized_corpus = [self.nori.do_analysis(doc)['termAtt'] for doc in tqdm(self.corpus,total=len(self.corpus))]
        # tokenized_corpus = [doc.split(' ') for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # with open('tokenized_corpus_nori.txt', 'w', encoding='utf-8', errors='ignore') as f:
        #     f.write(str(tokenized_corpus))

    def search(self, query, topk=10, mode=None):
        tokenized_query = self.nori.do_analysis(query)['termAtt']
        # tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_documents = self.bm25.get_top_n(tokenized_query, self.corpus, n=topk)

        if mode!='evaluate':
            print("Question :", query)
            print("Top Sentence :", top_documents)
            print("Score :", round(max(doc_scores), 3))

        return top_documents

    def evaluate(self, topk=10, mode=None):

        predict_lists = []

        for qc_pair in tqdm(self.QA, desc="Evaluate"):
            query = qc_pair["질문"]
            correct_context = qc_pair["문단"]
            top_documents = self.search(query, topk, mode=mode)

            bool_documents = [0]*topk
            for idx, document in enumerate(top_documents):
                bool_documents[idx] = int(correct_context == document)

            predict_lists.append(bool_documents)

            #[1 if correct_context==document else 0 for document in top_documents]

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


ir_base = IR_base()
#ir_base.search("임종석이 여의도 농민 폭력 시위를 주도한 혐의로 지명수배 된 날은?")
ir_base.evaluate(topk=10, mode='evaluate')
