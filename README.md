# BM25-KoSBERT

- [BM25](https://github.com/jaeyeongs/bm25) 알고리즘에 KoSentence-BERT 모델을 활용하여 재순서화 모듈을 추가한 정보 검색 엔진입니다.
- 형태소 분석기 pynori를 Tokenizer로 사용하였습니다.

> 모델에 대한 더 자세한 내용은 다음 [링크](https://github.com/jaeyeongs/research-develpoment/tree/main/Model/BM25-KoSBERT)를 참고해주세요. 

## Installation

```
git clone https://github.com/jaeyeongs/BM25-KoSBERT.git
pip install -r requirements.txt
```

## Train Models

### Dataset

- 모델 학습을 원하시면 아래 드라이브에서 Dataset을 다운 받으시면 됩니다.

![image](https://user-images.githubusercontent.com/87981867/207596578-c7b067d5-e4cb-4427-849d-05c577cecd8b.png)

*KorQuAD_STS 예시*

[[Dataset Download]](https://drive.google.com/file/d/1xJRoGUfVxl8iELXB998niiSw3NMcwzxe/view?usp=sharing)

### How to Train

- 학습 데이터는 STS 데이터 구조에 맞게 수정하였으며, 학습 방법은 아래와 같습니다.

```
with open('./MergeDataset/korquad_korpair_test.csv', 'rt', encoding='utf-8') as fIn: # 데이터 경로 설정
    lines = fIn.readlines()
    for line in lines:
        try:
            # s1, s2, score = line.split('\t') # 학습 데이터 구조 따라 변경 
            s1, s2, score = line.split(',')
            score = score.strip()
            score = float(score) / 5.0
            dev_samples.append(InputExample(texts= [s1,s2], label=score))
        except:
            continue
```
```
python train.py
```

## Pre-Trained Models
[Pre-trained Model Download](https://drive.google.com/drive/folders/1fLYRi7W6J3rxt-KdGALBXMUS2W4Re7II)

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
