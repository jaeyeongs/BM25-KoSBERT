# BM25-KoSBERT

- [BM25](https://github.com/jaeyeongs/bm25) 알고리즘에 KoSentence-BERT 모델을 활용하여 Reranking 모듈을 추가한 정보 검색 엔진
- 모델에 대한 더 자세한 내용은 [여기](https://github.com/jaeyeongs/research-develpoment/tree/main/Model/BM25-KoSBERT) 에서 확인

## Installation

```
git clone https://github.com/jaeyeongs/BM25-KoSBERT.git
pip install -r requirements.txt
```

## Train Models

## Pre-Trained Models

## Performance

|  | BM25+KoSentence-BERT(Kkma) | BM25+KoSentence-BERT(Nori) |
| :---: | :---: | :---: |
| MRR | 0.9216 | 0.9374 |
| Precision(k=10) | 0.9846 | 0.9889 |

## Application Examples

### Reranking

### Evaluate

## Reference

- [KoSentenceBERT-SKT](https://github.com/BM-K/KoSentenceBERT-SKT)
