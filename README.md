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

- pooling mode는 MEAN-strategy를 사용하였으며, 학습시 모델은 output 디렉토리에 저장 됩니다.
- 학습된 pt 파일은 아래 링크에 다운로드 가능합니다.

| **디렉토리** | **학습 방법** |
| :---: | :---: |
| training_nli | Only Train NLI |
| training_sts | Only Train STS |
| training_nli_sts | STS + NLI |

[Pre-trained Model Download](https://drive.google.com/drive/folders/1fLYRi7W6J3rxt-KdGALBXMUS2W4Re7II)

![image](https://user-images.githubusercontent.com/87981867/208038247-e2f671ba-61fd-4dc5-9165-8b372ca7b593.png)

각 폴더에 있는 result파일을 output 디렉토리에 넣으시면 됩니다.

ex) sts 학습 파일 사용시 위 드라이브에서 sts/result.pt 파일을 output/training_sts/0_Transformer에 넣으시면 됩니다.
output/training_sts/0_Transformer/result.pt

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

- **reranking.py**를 실행하시면 재순서화 모델 적용 결과를 출력해줍니다.
- 후보 문단 및 최종 순위 문단에 대한 score 값 확인이 가능합니다.

```
Question : 상고심 계류중에 사망한 영생교 교주의 사망원인은 무엇인가?
Top Sentence : ['결국 2004년 2월에 수원지방법원은 혐의를 완강하게 부인하던 교주 조희성과 처단조 행동대장 나경옥에게 사형을 선고했다. 다른 처단조 간부인 김진태는 무기징역 조모, 정모는 각각 징역 15,12년 형이 선고되었다. 그러나 교주 조희성과 간부들은 항소하였으며 원심과 달리 항소심에서 교주는 직접적으로 신도 살해를 지시하였는가에 대한 뚜렷한 증거가 없다는 이유로 살인에 대해서는 무죄를 선고받았으며 나경옥의 도피를 방조한 것에 대해서만 2년형을 선고받았다. 서울고법은 교주 조희성과는 대조적으로 처단조 간부 나경옥, 김진태, 조모, 정모에 대해서는 원심대로 판결하였다. 이후 교주 조희성은 상고심 계류중에 심근경색으로 사망하였다. 한편 같은 해 9월에 나머지 간부들은 1심과 2심 그대로 나경옥은 사형, 김진태는 무기징역, 조 모와 정 모에게는 각각 징역 15, 12년형을 선고한 원심이 그대로 확정되었다.', ...(중략)... '2010년 3월 26일, 백령도 근처 해상에서 대한민국 해군의 초계함인 PCC-772 천안이 침몰되는 사건이 일어났다. 대한민국 정부에서 발표한 이 사건의 공식 명칭은 "천안함 피격 사건"(天安艦被擊事件)이다. 북한 정찰총국 소행으로 보고 있다. 이 사건으로 대한민국 해군 병 40명이 사망했으며 6명이 실종되었다. 대한민국 정부는 천안함 침몰 원인을 규명할 민간·군인 합동조사단을 구성하였고, 대한민국을 포함한 오스트레일리아, 미국, 스웨덴, 영국 등 5개국에서 전문가 24여 명으로 구성된 합동조사단은 2010년 5월 20일 천안함이 조선민주주의인민공화국의 어뢰공격으로 침몰한 것이라고 발표하였다. 이러한 조사 결과 발표는 미국과 유럽 연합, 일본 외에 인도 등 비동맹국들의 지지를 얻어 국제 연합 안전보장이사회의 안건으로 회부되었으며. 안보리는 천안함 공격을 규탄하는 내용의 의장성명을 채택하였다. 그러나 조선민주주의인민공화국이 자신들과 관련이 전혀 없다고 주장하고, 중화인민공화국과 러시아가 반대하면서 조선민주주의인민공화국을 직접적으로 비난하는 내용에 이르지는 못했다. 조선민주주의인민공화국은 대한민국의 조사 결과에 대해 "특대형 모략극"이라며 사고 지점 근처에서 암초가 많다는 점을 들며 좌초한 것이라고 주장했다. 천안함의 침몰에서 인양, 조사 발표까지 대한민국 사회와 주변국의 관심을 끌었으며, 천안함의 침몰 원인을 규명하는 과정에서 언론과 각계 인사들을 통해 다수의 가설 또는 의혹들이 제기되기도 하였다. 이 사건으로 인해 남북간의 긴장이 고조되었으며, 대한민국에서는 침몰 원인에 대해 각기 다른 해석으로 갈등을 빚기도 했다.']
Score : 1.0
AFTER Sentence-BERT
Reranking Top Sentence : 결국 2004년 2월에 수원지방법원은 혐의를 완강하게 부인하던 교주 조희성과 처단조 행동대장 나경옥에게 사형을 선고했다. 다른 처단조 간부인 김진태는 무기징역 조모, 정모는 각각 징역 15,12년 형이 선고되었다. 그러나 교주 조희성과 간부들은 항소하였으며 원심과 달리 항소심에서 교주는 직접적으로 신도 살해를 지시하였는가에 대한 뚜렷한 증거가 없다는 이유로 살인에 대해서는 무죄를 선고받았으며 나경옥의 도피를 방조한 것에 대해서만 2년형을 선고받았다. 서울고법은 교주 조희성과는 대조적으로 처단조 간부 나경옥, 김진태, 조모, 정모에 대해서는 원심대로 판결하였다. 이후 교주 조희성은 상고심 계류중에 심근경색으로 사망하였다. 한편 같은 해 9월에 나머지 간부들은 1심과 2심 그대로 나경옥은 사형, 김진태는 무기징역, 조 모와 정 모에게는 각각 징역 15, 12년형을 선고한 원심이 그대로 확정되었다.
Reranking Score : 0.202
```

### Evaluate

- **evaluate.py**를 실행하면 모델 성능 지표인 [MRR](https://github.com/jaeyeongs/research-develpoment/tree/main/IR/metric/mrr) 값과 [Precision](https://github.com/jaeyeongs/research-develpoment/tree/main/IR/metric/precisionk) 값을 출력해줍니다.
- *Top10 Precision Score* 는 정답 문서가 1~10번째 나올 확률 값 입니다.

```
Evaluate: 100%|█████████████████████████████| 5774/5774 [07:57<00:00, 12.08it/s]
MRR Score : 0.9373546439705076
Top10 Precision Score : [0.9035330793210946, 0.9549705576723242, 0.9670938690682369, 0.9741946657429857, 0.9790439903013509, 0.9809490820921372, 0.9837201246969172, 0.9861447869760998, 0.9882230689296848, 0.9889158295808798]
```

## Reference

- [BM25](https://github.com/dorianbrown/rank_bm25)
- [KoSentenceBERT-SKT](https://github.com/BM-K/KoSentenceBERT-SKT)
- [Pynori](https://github.com/gritmind/python-nori)
