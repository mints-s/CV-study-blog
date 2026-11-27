---
layout: post
title: "표현 학습·대조 학습: 무엇을 같게 보고 무엇을 구별할까"
description: "두 view의 대응을 분류 문제로 바꾸고, NT-Xent의 분모·temperature·평가 조건을 직접 검산한다."
date: 2026-11-27 04:50:00 +09:00
category_label: 기초 개념
series_day: 12
topic: "REPRESENTATION & CONTRASTIVE LEARNING"
cover: /assets/day12-pairs.svg
tags: [representation, contrastive-learning, simclr]
reading_time: 10
---

> **1분 요약**
>
> 표현 학습은 이미지를 후속 과제에 유용한 특징으로 바꾸는 과정이다. 대조 학습의 한 방식은 같은 이미지의 두 변형을 positive로, 다른 이미지의 변형을 negative로 두고 올바른 짝을 찾게 한다. SimCLR에서는 encoder 특징 h를 projection head로 z에 옮겨 손실을 계산한다. 분모에는 positive도 들어가며 anchor 자신만 빠진다. 낮은 temperature는 큰 유사도에 더 집중하게 하지만 좋은 표현을 보장하지 않는다.
>
> **기억할 문장:** 증강과 짝의 정의가 무엇을 보존하고 무엇을 무시할지 정한다. 손실값과 후속 과제 성능은 별도로 확인해야 한다.

## 1. 전체 내용: 라벨이 없어도 학습 신호를 만든다

[Day 11]({{ '/notes/detection-segmentation-evaluation/' | relative_url }})에서는 예측의 평가 단위를 구분했다. 이번에는 그 예측을 만드는 특징을 어떻게 학습할지 살펴본다. 사람이 붙인 클래스 라벨이 부족해도, 같은 입력에서 만든 두 view라는 대응은 알 수 있다. 이 대응을 이용하면 후보 중 올바른 짝을 고르는 학습 문제를 만들 수 있다. 그러나 같은 원본이라는 사실이 두 crop의 의미까지 같다는 보장은 아니다. 학습된 특징은 별도의 라벨이 있는 과제로 평가해야 한다. 이 글은 SimCLR을 예로 든 기초 설명이며 전체 논문 리뷰나 대규모 학습 재현이 아니다.

**[원문·공식 코드]**, **[자체 검산]**, **[해석]·[아이디어]**를 구분한다. “라벨 없이 학습한다”는 표현은 여기서 사전학습에 클래스 라벨을 쓰지 않는다는 뜻이다. Positive의 생성 규칙도 학습 신호이며, 평가용 분류기의 훈련에는 라벨을 사용할 수 있다.

## 2. Encoder 특징과 손실용 특징을 구분한다

이미지 x의 두 view를 $v_i=t(x)$, $v_j=t'(x)$로 만든다. t와 t′는 증강 분포에서 각각 뽑은 변환이다. 공유 encoder f와 projection head g를 적용한다.

$$
h_i=f(v_i)\in\mathbb R^{d_h},\qquad
z_i=g(h_i)\in\mathbb R^{d_z},\qquad
u_i=\frac{z_i}{\|z_i\|_2}
$$

정규화 식은 0이 아닌 벡터에 대한 정의다. 구현에서는 작은 norm을 처리하는 epsilon도 확인해야 한다. h와 z의 차원은 같을 필요가 없다. 두 가지 view에 같은 f와 g를 사용하므로 별도 모델 둘을 학습하는 구조가 아니다.

**[원문]** SimCLR Figure 2와 §2.1은 f와 g를 분리하고, contrastive loss는 z에서 계산하며 후속 과제에는 h를 사용한다. [1] **[해석]** 손실에 직접 맞춘 마지막 벡터만이 좋은 특징이라고 가정하지 말아야 한다. 평가할 층 자체도 실험 조건이다.

| 표현 | 역할 | 확인할 질문 |
| --- | --- | --- |
| 입력 view v | 증강된 관측 | 물체의 의미가 보존되는가? |
| Encoder 특징 h | 후속 과제의 입력 후보 | 어떤 층·pooling을 사용했는가? |
| Projection z | 손실을 적용할 공간 | 차원과 head 구조가 무엇인가? |
| 정규화 u | cosine 비교 | 정규화 축과 epsilon이 무엇인가? |

## 3. 배치 안의 짝을 분류 문제로 읽는다

N개의 원본에서 각각 두 view를 만들면 2N개의 벡터가 생긴다. 이 글은 같은 원본의 view를 연속으로 배치한다. N=2이면 순서는 A₁, A₂, B₁, B₂다. A₁의 positive는 A₂이며 B₁·B₂가 negative다. A₁ 자체는 비교 후보에서 제외한다.

<figure>
<img src="{{ '/assets/day12-pairs.svg' | relative_url }}" alt="4개 view A1 A2 B1 B2의 유사도 행렬에서 대각선은 제외하고 A1-A2, A2-A1, B1-B2, B2-B1은 positive로 표시한다. 나머지 8개 항은 negative다.">
<figcaption>그림 1. 자체 작성한 N=2의 후보 행렬이다. 행은 anchor, 열은 비교 후보이며 칸의 숫자는 5절의 고정 단위 벡터 내적이다. 청록색은 positive, 회색 대각선은 제외, 노란색은 negative다. 원문 Figure의 복제가 아니다.</figcaption>
</figure>

**[원문]** 원래 SimCLR의 한 anchor에는 positive 1개와 negative 2N−2개가 있다. 양쪽 방향을 모두 손실에 포함한다. [1] 여기서 negative는 다른 클래스라는 정답이 아니라 다른 원본에서 왔다는 구성 규칙이다. 같은 클래스의 다른 이미지가 negative가 될 수 있다.

정규화된 벡터의 cosine similarity는 $s_{ik}=u_i^\top u_k$이며 범위는 −1부터 1이다. Positive index를 p(i), temperature를 $\tau>0$로 두면 다음과 같다.

$$
\ell_i=-\log\frac{\exp(s_{i,p(i)}/\tau)}{\sum_{k\ne i}\exp(s_{ik}/\tau)},\qquad
L=\frac1{2N}\sum_{i=1}^{2N}\ell_i
$$

분모는 **positive와 negative를 모두** 포함한다. Positive를 분모에서 빼면 후보 확률이라는 해석이 깨진다. 반대로 자기 자신을 넣으면 정규화된 벡터의 자기 내적 1이 강력한 경쟁자가 된다. Positive index를 배치 순서에 맞게 만드는 것이 중요하다.

**[공식 문서]** CrossEntropyLoss는 정규화 전 logits와 정답 index를 입력받을 수 있다. [4] 각 행에서 자기 칸을 제외한 $s/\tau$를 logits로, positive 칸을 정답으로 보면 위 손실은 교차 엔트로피다. 이미 softmax한 확률을 다시 logits 입력으로 넣는 것과는 다르다.

### 공식 코드와 비교할 때 생기는 두 차이

**[공식 코드]** 확인한 `objective.py`는 첫 view 묶음과 두 번째 view 묶음으로 텐서를 나누고, 같은 묶음의 대각선을 큰 음수로 마스킹한다. 이 글의 연속 pair 배치와 index 순서가 다르다. 또 `loss_a + loss_b`로 방향별 평균을 더하므로, 동일 데이터·가중치라면 위의 전체 2N 평균과 비교해 상수 2배 차이가 난다. [3] 손실 로그의 절댓값을 비교하기 전에 reduction을 맞춰야 한다.

## 4. Temperature는 유사도 차이를 확대한다

유사도를 tau로 나누면 같은 차이도 작은 tau에서 더 큰 logit 차이가 된다. 그러나 모든 경우에 손실을 낮추는 장치가 아니다. Positive가 가장 큰 유사도를 가진다면 작은 tau가 그 확률을 올리기 쉽다. Negative가 더 크다면 반대로 오답 후보에 확률이 집중된다.

**[자체 유도]** 후보 확률을 $q_{ik}$라 하면 유사도에 대한 미분은 다음과 같다.

$$
\frac{\partial\ell_i}{\partial s_{ik}}=
\frac{q_{ik}-\mathbf1[k=p(i)]}{\tau}\quad(k\ne i)
$$

이는 유사도 변수를 독립 입력으로 보았을 때의 미분이다. 실제 encoder 파라미터의 gradient는 정규화·내적·신경망의 연쇄 미분을 거친다. Gradient에 1/tau가 나타나지만 q도 tau에 따라 바뀌므로 gradient 크기가 항상 단순 반비례하는 것은 아니다.

<details markdown="1">
<summary>수치적으로 안정적인 계산</summary>

$a_k=s_{ik}/\tau$라 하면 $\ell_i=\operatorname{logsumexp}(a)-a_{p(i)}$다. 최대값 m을 빼서 $\operatorname{logsumexp}(a)=m+\log\sum_k\exp(a_k-m)$으로 계산하면 큰 지수값으로 인한 overflow 위험을 줄인다. 마스킹은 logsumexp 전에 적용한다. 본문의 검산 코드는 후보 목록에서 자기 index를 제거한다.

</details>

## 5. 고정 벡터 실험: 학습 없이 손실을 확인한다

**[자체 실험]** 다음 네 벡터는 이미 길이가 1이다. Positive는 (1,2), (3,4)이며 양방향을 모두 사용한다.

$$
u_1=(1,0),\quad u_2=(0.8,0.6),\quad
u_3=(0,1),\quad u_4=(-0.6,0.8)
$$

Python 표준 라이브러리의 double precision 부동소수점 연산을 사용했다. Encoder·증강·학습·난수는 없다. 첫 anchor의 후보 유사도는 (0.8, 0, −0.6)이다. tau=0.5이면 logits는 (1.6, 0, −1.2)이고 positive 확률은 0.791950, 손실은 0.233257이다.

| tau | 첫 anchor의 positive 확률 | 첫 anchor 손실 | 네 anchor 평균 L |
| --- | --- | --- | --- |
| 1.0 | 0.589648 | 0.528229 | 0.673577 |
| 0.5 | 0.791950 | 0.233257 | 0.430190 |
| 0.1 | 0.999664 | 0.000336 | 0.063780 |

두 번째 anchor는 negative 중 하나와 유사도 0.6을 가지므로 더 어렵다. tau=0.5에서 그 손실은 0.627123이다. 한 anchor의 쉬운 결과를 전체 평균으로 쓰면 안 된다.

**반례 검산:** 첫 anchor의 negative 유사도 하나를 0.9로 바꾸면, tau=0.1에서 positive 확률은 0.268941이고 손실은 1.313262다. 여기서는 한 행의 logits만 바꾼 별도 예제이며 위 네 벡터를 그대로 사용한 결과가 아니다. 작은 tau가 “항상 좋은 값”이라는 결론은 성립하지 않는다.

**구현 오류 검산:** tau=0.5의 첫 anchor에서 자기 내적 1을 분모에 잘못 넣으면 손실은 1.013247이 된다. 올바른 0.233257과 크게 다르다. Positive 유사도의 미분은 −0.416101이며 중앙 유한차분(step=1e−6)과 일치하는지 확인했다.

[전체 검증 코드]({{ '/assets/day12-verify.py' | relative_url }})는 `python day12-verify.py`로 실행한다. 공식 TensorFlow 학습 코드를 실행한 결과가 아니며, 데이터셋 성능이나 SimCLR 재현 정확도를 보고하는 실험도 아니다.

## 6. 붕괴와 false negative를 구분한다

모든 입력이 같은 단위 벡터로 표현되면 후보 유사도가 모두 같다. N=2에서 positive 확률은 1/3, 손실은 $\log3=1.098612$다. 위 분리된 네 벡터의 tau=0.5 평균보다 크다. **[자체 검산]** 이 목적함수에서 모든 벡터가 같아지는 것이 좋은 해답은 아니라는 비교다. 이것만으로 실제 최적화가 모든 형태의 붕괴를 반드시 피한다고 증명하지는 않는다.

False negative는 다른 원본이지만 의미상 비슷한 입력을 negative로 다루는 문제다. 예를 들어 서로 다른 두 고양이 사진은 같은 원본이 아니므로 이 구성에서는 negative다. 클래스 분류에는 공통점을 보존하는 편이 유용할 수 있지만 instance 구별에는 차이도 필요하다. 어떤 유사성을 학습할지는 후속 과제와 연결해 판단해야 한다.

증강도 같은 문제를 만든다. 색이 과제의 정답을 결정하는데 색 정보를 강하게 지우면 필요한 구분이 사라질 수 있다. 물체가 작은 이미지에서 두 crop이 각각 물체와 배경만 담으면, 같은 원본이라는 규칙이 의미상 같은 내용까지 보장하지 못한다. **[해석]** 증강은 단순히 학습 데이터를 늘리는 도구를 넘어 불변성의 가정을 설정한다.

## 7. 원문 실험을 읽을 때 확인할 조건

**[원문 부록]** SimCLR Supplement A는 ResNet-50을 100 epochs 학습한 조건에서 Gaussian blur를 넣기 전후의 linear evaluation top-1을 63.2%와 64.5%로 보고한다. Blur 적용 확률은 50%, sigma 범위는 [0.1,2.0]이다. 같은 절은 crop·flip·색 변형의 세부 설정도 제공한다. [2] **[해석]** 이는 해당 학습 조건의 결과다. 다른 해상도나 색 구분 과제에도 같은 향상을 보장하지 않으며, 본 글에서 재현하지 않았다.

| 평가 방식 | 학습하는 부분 | 읽을 수 있는 것 |
| --- | --- | --- |
| Contrastive pair 정확도 | 사전학습 중의 짝 선택 | 현재 후보 집합에서 짝을 찾는 능력 |
| Linear probe | encoder를 고정하고 선형 분류기만 학습 | 특징의 선형 분리 가능성 |
| Fine-tuning | encoder의 일부 또는 전체도 갱신 | 정해진 적응 절차 후의 성능 |
| 검출·분할 전이 | 해당 head와 정해진 backbone 범위 | 위치·픽셀 과제의 전이 성능 |

첫 행의 정확도와 나머지는 같은 지표가 아니다. Linear probe에서도 훈련 데이터·특징 층·정규화·학습률 탐색·평가 split이 필요하다. “Encoder 고정”에는 파라미터 gradient뿐 아니라 BatchNorm 통계 갱신 여부도 포함해 기록한다. Fine-tuning에서는 갱신 범위를 명시한다.

[앞선 평가 글]({{ '/notes/detection-segmentation-evaluation/' | relative_url }})과 연결하면, 분류 probe가 좋다는 사실만으로 검출 AP나 분할 mIoU가 높다고 결론 내릴 수 없다. 전역 특징의 불변성과 세밀한 위치 정보 보존은 과제마다 요구가 다르다.

## 8. 한계와 검증 가능한 확장 아이디어

| 흔한 해석 | 확인할 조건 |
| --- | --- |
| 라벨이 없으므로 가정도 없다 | view 생성·positive·negative 규칙 |
| 낮은 손실이면 더 좋은 표현이다 | tau, 후보 수, reduction, 별도 probe |
| 배치가 커지면 항상 좋아진다 | 학습률·총 update·메모리·false negative |
| 학습 후 z가 반드시 가장 좋다 | h와 z를 동일 probe 조건으로 비교 |
| 불변성이 클수록 좋다 | 후속 과제가 구별해야 하는 변화 |

**[아이디어 · 미실행]** 색 불변성을 강화하면 형태 기반 분류에는 도움이 되지만 색 판별에는 손해가 있는지 검사한다.

- **가설:** 같은 계산 예산에서 강한 색 변형은 색 정보의 선형 판별 가능성을 낮추고, 형태 분류에는 다른 영향을 준다.
- **Baseline·비교:** 같은 encoder와 대조 손실에 약한 색 jitter와 강한 색 jitter를 적용한 두 사전학습이다. Crop·blur 등 나머지는 고정한다.
- **최소 실험:** 형태 4종과 색 4종을 독립 배정한 합성 이미지 8,000장으로 사전학습하고, 별도 생성한 2,000장으로 형태·색 각각의 linear probe를 훈련한다. 또 다른 2,000장을 최종 평가에 사용한다.
- **통제:** 배경·위치·크기 분포, encoder 초기화 seed 쌍, 배치 크기·tau·총 update 수·optimizer·probe 탐색 예산을 맞춘다. 3개 seed를 사용하고 렌더링 파라미터 조합의 split 간 중복을 제거한다.
- **지표:** 형태 top-1, 색 top-1, h의 차원별 분산과 공분산 고유값 분포를 기록한다. 평균과 seed별 결과를 함께 제시한다.
- **기각 기준:** 색 판별 저하가 seed에 걸쳐 나타나지 않으면 색 정보 소실 가설을 지지하지 않는다. 형태 성능도 떨어지면 유용한 불변성보다 정보 손실 가능성을 검토한다.

아직 실행하지 않은 제안이다. 합성 도형 결과만으로 자연영상에서의 우열을 일반화하지 않는다. 이 실험은 데이터와 증강의 가정을 분리해 보는 최소 점검이다.

## 9. 복습

1. N개의 원본에서 두 view씩 만들 때 anchor 하나의 분모에는 몇 개 후보가 들어가는가?
2. Positive를 분모에서 빼거나 자기 자신을 넣으면 무엇이 달라지는가?
3. 작은 tau에서 손실이 줄었다는 결과만으로 encoder가 좋아졌다고 말할 수 있는가?

<details markdown="1">
<summary>답</summary>

1. 2N−1개다. Positive 1개와 negative 2N−2개이며 anchor 자체는 없다.
2. 원래의 짝 분류 문제가 달라진다. Positive는 후보에 포함되고 자기 자신은 제외되어야 한다.
3. 아니다. 고정 벡터에서도 loss는 바뀐다. 같은 평가 조건의 linear probe나 후속 과제를 따로 확인해야 한다.

</details>

## 출처와 다음 읽기

1. Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton, **A Simple Framework for Contrastive Learning of Visual Representations**, ICML 2020, PMLR 119, [정식 본문](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf). §2.1·Figure 2·Algorithm 1의 구조와 손실을 확인했다.
2. [공식 Supplement](https://proceedings.mlr.press/v119/chen20j/chen20j-supp.pdf), Appendix A, PDF 1–2쪽. 증강 설정과 blur 비교 조건을 확인했다.
3. [공식 SimCLR objective.py](https://github.com/google-research/simclr/blob/master/objective.py), `add_contrastive_loss`의 정규화·self mask·방향별 reduction을 확인했다.
4. [PyTorch 2.15 CrossEntropyLoss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html), logits와 class-index target 정의를 확인했다.

확인일: 2026-11-27. 도식과 고정 벡터 예제는 자체 작성했다. 학습 성능 재현과 제안 실험은 수행하지 않았다.

[이전: Detection·Segmentation과 평가]({{ '/notes/detection-segmentation-evaluation/' | relative_url }}) · [전체 학습 지도]({{ '/roadmap/' | relative_url }})

**다음 질문:** 3D 공간의 밀도와 색을 한 픽셀로 모으는 규칙은 무엇일까? 다음 글은 [Day 13: Ray·Volume Rendering]({{ '/notes/ray-volume-rendering/' | relative_url }})이다.

