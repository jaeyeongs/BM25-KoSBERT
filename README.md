# BM25-KoSBERT

- [BM25](https://github.com/jaeyeongs/bm25) 알고리즘에 KoSentence-BERT 모델을 활용하여 Reranking 모듈을 추가한 정보 검색 엔진입니다.
- 형태소 분석기 pynori를 Tokenizer로 사용하였습니다.

> 모델에 대한 더 자세한 내용은 다음 [링크](https://github.com/jaeyeongs/research-develpoment/tree/main/Model/BM25-KoSBERT)를 참고해주세요. 

## Installation

```
git clone https://github.com/jaeyeongs/BM25-KoSBERT.git
pip install -r requirements.txt
```

## Train Models

## Performance

- 모델 평가는 모두 KorQuAD_v1.0_dev Dataset으로 진행하였습니다.
- 각 Dataset 별로 훈련에 모델을 가지고 평가한 결과입니다.

|  | **STS + NLI** | **KorQuAD_STS** | **PairedQuestion_v.2** | **KorQuAD_v1.0 + PairedQuestion_v.2** |
| :---: | :---: | :---: | :---: | :---: |
| MRR | 0.727 | 0.765 | 0.832 | 0.835 |
| Precision(k=10) | 0.899 | 0.898 | 0.899 | 0.898 |

- 모델 고도화를 위해 형태소 분석기(pynori) 모듈을 추가한 후 성능 평가한 결과입니다.

|  | **BM25 + KoSentence-BERT** |
| :---: | :---: |
| MRR | 0.9374 |
| Precision(k=10) | 0.9889 |

## Application Examples

### Reranking

### Evaluate

- evaluate.py를 실행하면 모델 성능 지표인 [MRR](https://github.com/jaeyeongs/research-develpoment/tree/main/IR/metric/mrr) 값과 [Precision](https://github.com/jaeyeongs/research-develpoment/tree/main/IR/metric/precisionk) 값을 출력
- *Top10 Precision Score* 는 정답 문서가 1~10번째 나올 확률 값

```
Evaluate: 100%|█████████████████████████████| 5774/5774 [07:57<00:00, 12.08it/s]
MRR Score : 0.9373546439705076
Top10 Precision Score : [0.9035330793210946, 0.9549705576723242, 0.9670938690682369, 0.9741946657429857, 0.9790439903013509, 0.9809490820921372, 0.9837201246969172, 0.9861447869760998, 0.9882230689296848, 0.9889158295808798]
```

## Reference

- [BM25](https://github.com/dorianbrown/rank_bm25)
- [KoSentenceBERT-SKT](https://github.com/BM-K/KoSentenceBERT-SKT)
- [Pynori](https://github.com/gritmind/python-nori)
