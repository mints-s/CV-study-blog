---
layout: post
title: "Attention의 Q, K, V: 무엇을 비교하고 무엇을 가져오는가"
description: "Q·K의 점수와 V의 내용을 분리하고, 행렬 차원·softmax·마스크를 작은 수치 예제로 확인한다."
date: 2026-11-24 05:00:00 +09:00
category_label: 기초 개념
series_day: 9
topic: "ATTENTION · QKV"
cover: /assets/day09-attention.svg
tags: [attention, qkv, transformer, foundations]
reading_time: 10
---

> **1분 요약**
>
> Attention은 현재 위치의 **Q(query)**와 후보들의 **K(key)**를 비교해 가중치를 만든 뒤, 후보의 **V(value)**를 그 비율로 섞는다. Q·K는 선택 기준이고 V는 가져올 내용이다. Scaled dot-product attention은 점수를 $\sqrt{d_k}$로 나누고 **key 축으로 softmax**를 적용한다. Self-attention도 Q=K=V라는 뜻은 아니다. 같은 입력에서 서로 다른 학습 투영을 만들 수 있다.
>
> **기억할 문장:** attention 행렬은 출력 특징이 아니다. 가중치 A와 내용 V를 곱한 AV가 출력이다.

## 1. 전체 내용: 기하적 연결에서 학습된 정보 선택으로

[Day 8]({{ '/notes/sfm-bundle-adjustment/' | relative_url }})에서는 관측 연결을 바탕으로 카메라와 점을 보정했다. 이제 관심을 특징 벡터 사이의 정보 교환으로 옮긴다. 영상 패치나 특징점마다 벡터가 있을 때, 현재 위치에 필요한 정보를 다른 위치에서 모으는 연산이 필요하다. Attention은 입력에 따라 달라지는 가중치로 이를 수행한다. 단, 큰 attention 가중치가 같은 물리적 점의 대응을 증명하는 것은 아니다. 학습 목적·표현·위치 정보가 무엇인지 함께 살펴야 한다.

이 글은 attention 연산의 기초를 다룬다. Transformer 전체 구조와 위치 표현은 다음 회차로 연결한다. **[원문]·[공식 코드]**의 확인 내용과 **[자체 검산]**, **[해석]·[아이디어]**를 구분한다. 별도 표기가 없으면 단일 head, dropout 없음, bias 없음이며 한 행을 한 token으로 저장한다.

## 2. 세 역할: 검색 기준과 내용은 다르다

입력 특징을 $X\in\mathbb R^{n\times d_{model}}$이라 하면 self-attention의 한 head는 다음처럼 만들 수 있다.

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

$W_Q,W_K\in\mathbb R^{d_{model}\times d_k}$, $W_V\in\mathbb R^{d_{model}\times d_v}$다. 학습되는 것은 이 투영 행렬 등의 파라미터이며 Q·K·V 자체는 입력에 따라 바뀌는 활성값이다. 이전 기하 글의 K는 카메라 내부 행렬이었지만, **이 글의 K는 key 행렬**이다.

| 역할 | 읽는 질문 | 혼동하기 쉬운 점 |
| --- | --- | --- |
| Q | 이 위치는 어떤 정보를 찾는가? | 사람이 작성한 질문 문장일 필요가 없다 |
| K | 각 후보를 어떤 기준으로 비교하는가? | 데이터베이스의 고정 ID가 아니다 |
| V | 선택 비율에 따라 어떤 내용을 전달하는가? | K와 같은 벡터일 필요가 없다 |

**[해석]** “검색”은 연산을 이해하기 위한 비유다. 실제로는 연속 벡터의 내적과 미분 가능한 가중합이다. [Day 2]({{ '/notes/loss-gradient-backprop/' | relative_url }})의 역전파를 통해 투영이 학습되며, 각 차원에 인간이 읽을 수 있는 의미가 자동으로 붙는 것은 아니다.

Self-attention은 Q·K·V의 **원천이 같은 입력**인 경우다. Cross-attention에서는 Q의 원천과 K·V의 원천이 다르다. 예를 들어 첫 영상 특징이 Q를 만들고 두 번째 영상 특징이 K·V를 만들 수 있다. Q의 개수와 K·V의 개수도 달라도 된다. 다만 K와 V는 같은 후보 순서로 짝지어야 한다.

## 3. 수식과 차원: softmax는 어느 방향인가

Q의 개수를 m, key/value의 개수를 n이라 두면 다음과 같다.

$$
S=\frac{QK^\top}{\sqrt{d_k}},\qquad
A_{ij}=\frac{\exp(S_{ij})}{\sum_{l=1}^n\exp(S_{il})},\qquad
O=AV
$$

| 텐서 | 차원 | 의미 |
| --- | --- | --- |
| Q | $m\times d_k$ | m개 query |
| K | $n\times d_k$ | n개 key |
| V | $n\times d_v$ | key와 짝을 이루는 내용 |
| S, A | $m\times n$ | query별 후보 점수와 가중치 |
| O | $m\times d_v$ | query마다 모은 결과 |

**[원문]** Vaswani 등의 §3.2는 query–key 비교로 value의 가중합을 만들고, scaled dot-product와 multi-head를 정의한다. [1] 위 표는 그 식을 m과 n이 다른 경우까지 읽기 쉽게 펼친 것이다.

각 행의 합은 1이고 각 원소는 음수가 아니다. 열의 합이 1일 필요는 없으며, 서로 다른 query가 같은 후보에 높은 가중치를 줄 수 있다. Dropout을 적용하기 전의 가중치에 대한 성질이다. 이 조건의 O 한 행은 V 행들의 볼록 결합이지만, 뒤의 출력 투영·잔차 연결·FFN까지 포함한 결과가 원래 V의 볼록 결합일 필요는 없다.

Softmax는 QK의 원소별 정규화가 아니라 **한 query의 모든 후보를 함께 비교하는 정규화**다. 후보 하나를 추가해도 분모가 바뀌어 기존 후보의 비중이 달라진다. 유사한 후보가 여러 번 복제되면 전체 가중치가 그쪽으로 몰릴 수도 있다.

## 4. 왜 나누는 값이 $\sqrt{d_k}$인가

**[원문]** §3.2.1의 각주에서는 q와 k 성분이 독립이고 평균 0·분산 1이라는 가정 아래 내적의 분산이 $d_k$가 됨을 설명한다. 점수가 커져 softmax가 포화되는 문제를 완화하려고 $\sqrt{d_k}$로 나눈다. [1]

이는 각 벡터를 단위 길이로 만드는 연산이 아니다. 따라서 cosine similarity와 같지 않다. 학습 후의 실제 성분들이 항상 독립·분산 1이라는 보장도 없다. 이 예제에서 Q와 K를 각각 4배로 만들면 분모는 같지만 내적은 16배가 된다.

**[공식 코드]** Tensor2Tensor의 일반 multihead 경로는 head별 깊이로 q를 먼저 스케일한다. `dot_product_attention` 내부에서는 QKᵀ, bias, softmax, dropout, V와의 곱 순서를 확인했다. [2] 함수 안에 나눗셈이 없다는 이유만으로 스케일링이 빠졌다고 결론 내리면 안 된다. 호출 전 처리를 함께 읽어야 한다.

## 5. 수치 예제: query 2개가 후보 3개를 읽는다

**[자체 검산]** 학습된 모델이 아니라 계산용으로 정한 Q·K·V다. $m=2,n=3,d_k=d_v=2$, NumPy 2.3.5, float64, 난수·dropout·출력 투영은 없다.

$$
Q=\begin{bmatrix}1&0\\0&1\end{bmatrix},\quad
K=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad
V=\begin{bmatrix}10&0\\0&10\\4&4\end{bmatrix}
$$

첫 query의 내적은 (1,0,1)이다. $\sqrt2$로 나누면 (0.707107,0,0.707107)이고, softmax 후에는 약 (0.401112,0.197776,0.401112)가 된다. 출력은 다음처럼 계산한다.

$$
O_1=0.401112(10,0)+0.197776(0,10)+0.401112(4,4)
\approx(5.615569,3.582207)
$$

<figure>
<img src="{{ '/assets/day09-attention.svg' | relative_url }}" alt="행이 query, 열이 key인 2×3 가중치 행렬과 3×2 value 행렬의 곱으로 2×2 출력이 만들어진다. 첫 query의 가중치는 0.401112, 0.197776, 0.401112다.">
<figcaption>그림 1. 직접 계산한 A·V·O다. 가중치의 행은 query, 열은 key/value 후보다. 숫자는 소수 여섯 자리로 반올림했으며 계산은 반올림 전 값으로 수행했다. 원문 Figure를 복제한 그림이 아니다.</figcaption>
</figure>

| 설정 | 첫 query의 가중치 | 첫 query의 출력 |
| --- | --- | --- |
| 기본 scaled attention | (0.401112, 0.197776, 0.401112) | (5.615569, 3.582207) |
| 나눗셈 제거 | (0.422319, 0.155362, 0.422319) | 스크립트의 가중치 비교 대상 |
| 세 번째 후보 금지 | (0.669762, 0.330238, 0) | (6.697615, 3.302385) |

첫 번째와 세 번째 key는 첫 query에 같은 점수를 받지만 V는 다르다. **같은 비교 점수와 같은 전달 내용은 별개의 조건**이다. 또한 기본 출력은 V의 어느 한 행을 그대로 복사한 것이 아니라 여러 행을 혼합한 값이다.

Q·K를 각각 4배로 한 추가 검산에서는 첫 행의 두 번째 가중치가 약 $6.1022\times10^{-6}$으로 작아졌다. 첫 번째와 세 번째는 동점이라 각각 약 0.499997이다. 점수가 커진다고 반드시 하나의 후보만 선택되는 것은 아니다. 이 결과는 스케일의 산술 효과이며 학습 안정성이나 정확도 실험이 아니다.

## 6. 마스크: 읽지 못할 후보를 softmax 전에 제외한다

허용 위치는 0, 금지 위치는 $-\infty$인 additive mask M을 두면 $A=\operatorname{softmax}(S+M)$이다. 위 표에서는 세 번째 열을 금지했다. 나머지 두 후보의 합이 다시 1이 되면서 출력도 달라졌다. Softmax 후 세 번째 값만 0으로 바꾸고 재정규화하지 않는 것과 다르다.

Padding mask는 유효하지 않은 토큰을, causal mask는 미래 위치를 읽지 못하게 한다. 예제의 마스크는 모든 query에서 세 번째 후보를 금지하는 것으로, causal mask가 아니다. 각 행에 허용된 후보가 하나도 없다면 수학적 softmax의 분모가 0이 된다. 구현마다 특수 처리가 다를 수 있으므로 이를 명시적으로 다뤄야 한다. 첨부 코드는 그런 행을 assertion으로 거부한다.

**[공식 문서]** PyTorch의 `scaled_dot_product_attention`에서 boolean mask의 True는 **참여 허용**이다. `MultiheadAttention.key_padding_mask`의 True와 의미가 반대이므로 그대로 재사용하지 않는다. 이 함수는 지정한 `dropout_p`에 따라 dropout을 적용하므로 평가 시에는 0.0을 전달해야 한다. [3]

<details markdown="1">
<summary>재현 코드와 검증 항목</summary>

[전체 검증 스크립트]({{ '/assets/day09-verify.py' | relative_url }})를 NumPy 2.3.5 환경에서 `python day09-verify.py`로 실행한다. 아래는 핵심 연산이다.

~~~python
scores = Q @ K.T / np.sqrt(Q.shape[-1])
# mask가 있다면 True만 허용한다.
scores = np.where(mask, scores, -np.inf)
weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
weights /= weights.sum(axis=-1, keepdims=True)
output = weights @ V
~~~

최댓값을 빼는 것은 지수 계산을 안정화하며 정확한 산술에서는 softmax를 바꾸지 않는다. 전체 코드에서는 마스크가 없는 경로도 처리한다. 행 합 1, 금지 가중치 0, K·V 동시 순열의 출력 불변성, self-attention의 순열 등변성을 검사했다. PyTorch·Tensor2Tensor를 실행한 것은 아니다.

</details>

## 7. Multi-head와 위치 정보의 경계

Multi-head는 head마다 다른 투영으로 정보를 모으고 결과를 이어 붙인 뒤 출력 투영한다. [1]

$$
H_h=\operatorname{Attention}(XW_Q^{(h)},XW_K^{(h)},XW_V^{(h)}),\qquad
O_{multi}=\operatorname{Concat}(H_1,\ldots,H_H)W_O
$$

head 수를 H라 하면 이어 붙인 결과는 $n\times(Hd_v)$이고, $W_O\in\mathbb R^{Hd_v\times d_{model}}$일 때 최종 차원은 $n\times d_{model}$이다. 각 head가 반드시 “윤곽”, “색”, “동작”처럼 정해진 역할을 담당한다는 뜻은 아니다.

**[자체 검산·해석]** 위치 정보와 마스크가 없는 self-attention f에서 토큰을 순열 행렬 P로 재배열하면 $f(PX)=Pf(X)$다. 입력 순서를 바꾸면 출력도 같은 순서로 바뀌는 **등변성**이며, 모든 출력이 그대로인 불변성과 다르다. 첨부 코드에서 3개 토큰의 순서를 (3,1,2)로 바꾸어 확인했다. 같은 Q를 두고 K·V만 함께 재배열하면 출력은 그대로라는 성질도 따로 확인했다.

따라서 토큰의 내용만으로 계산하는 attention에 “왼쪽 위”나 “먼저 발생한 시점”의 의미가 저절로 주어지지는 않는다. 위치 정보가 특징에 들어 있거나 mask·bias로 제공된다면 조건이 달라진다. 다음 회차에서는 이를 Transformer의 잔차 연결·정규화·FFN과 함께 묶는다.

## 8. CV에서 읽을 때의 한계와 시사점

| 확인 사항 | 이유 |
| --- | --- |
| query와 memory의 출처 | self/cross 여부와 출력의 위치 기준이 달라진다 |
| 토큰화 단위 | 픽셀·패치·특징점에 따라 n과 정보 손실이 달라진다 |
| 점수와 출력의 구분 | A만 보아서는 V의 크기·방향과 후속 투영의 효과를 알 수 없다 |
| 위치·기하 제약 | 큰 가중치가 올바른 3D 대응을 보장하지 않는다 |
| 계산·메모리 조건 | 직접 구성한 A는 $m\times n$이며, dense self-attention은 n²개의 비교를 포함한다 |

예를 들어 14×14 패치 196개와 28×28 패치 784개를 비교하면 토큰 수는 4배, 단일 head 가중치 원소 수는 38,416개에서 614,656개로 16배다. 이는 A를 명시적으로 저장할 때의 원소 수 계산이며, 최적화된 커널의 실제 메모리나 실행 시간을 측정한 결과가 아니다. QKV 투영 비용과 dtype·batch·head 수도 별도다.

**[원문 확인]** 원문의 부록 Figure 3–5는 encoder layer 5의 언어 attention 시각화다. [1] **[해석]** 선이 강하게 나타난다는 관찰만으로 해당 연결이 최종 판단의 인과적 설명이라고 단정하지 않는다. V와 다른 경로의 영향을 확인하고, 연결을 바꾸거나 제거했을 때 출력이 어떻게 변하는지 따로 실험해야 한다.

## 9. 검증 가능한 확장 아이디어

**[아이디어 · 미검증]** 두 영상 사이의 cross-attention에 기하적 후보 제한을 넣으면 반복 무늬에서 오대응을 줄일 수 있을까?

- **가설:** 보정과 상대 자세가 충분히 정확할 때 epipolar 거리 기반 mask가 잘못된 후보를 줄인다.
- **비교:** 같은 고정 특징·QKV에서 제한 없는 attention과 1·2·4 pixel 허용 폭의 mask를 비교한다. 빈 후보 행은 별도 실패로 집계한다.
- **통제:** 같은 영상 쌍, 특징점, K, 상대 자세, 해상도, 평가 정답을 사용한다. 온도나 특징 투영을 조건별로 재학습하지 않는다.
- **최소 실험:** 정답 대응이 있는 합성 100쌍을 만들고 반복 무늬와 가림 여부로 나눈다. 별도로 상대 회전 추정에 0·0.5·1도 오차를 가해 기하 불확실성을 평가한다.
- **지표:** 참 대응의 후보 유지율, 가림 없는 query의 top-1 대응 정확도, 빈 행 비율을 함께 기록한다. 최대 가중치만으로 가림 검출 성능을 주장하지 않는다.
- **기각 기준:** 정확도 상승이 참 대응의 대량 탈락을 숨기거나 작은 자세 오차에서 사라지면 일반적인 개선 주장을 기각한다.

이 실험은 수행하지 않았다. 작은 QKV 예제는 기하 마스크의 실영상 효과를 입증하지 않는다.

## 10. 복습

1. A의 행 합은 1인데 열 합이 1일 필요가 없는 이유는 무엇인가?
2. Self-attention이라는 이름이 Q=K=V를 뜻하지 않는 이유는 무엇인가?
3. 위 예제에서 첫 번째와 세 번째 key의 점수가 같아도 두 후보가 동일한 정보를 주지 않는 이유는 무엇인가?

<details markdown="1">
<summary>답</summary>

1. query마다 후보 key 축으로 softmax를 따로 계산하기 때문이다. 여러 query가 같은 후보를 동시에 읽을 수 있다.
2. 동일한 입력 X에서도 서로 다른 투영 행렬 WQ·WK·WV를 사용하기 때문이다.
3. 대응 V가 (10,0)과 (4,4)로 다르기 때문이다. 비교 기준과 전달 내용은 별개다.

</details>

## 출처와 다음 읽기

1. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin, **Attention Is All You Need**, NIPS 2017. [arXiv v7 HTML](https://arxiv.org/html/1706.03762v7) · [PDF](https://arxiv.org/pdf/1706.03762v7). 확인 버전은 2023-08-02 v7이며 §3.2, 식 (1), Figure 2와 부록 Attention Visualizations(PDF 13–15쪽)을 확인했다. 번역 성능 전체 리뷰가 아니라 기초 연산 설명에 사용했다.
2. [공식 Tensor2Tensor common_attention.py](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py), 확인 blob `11ce57fcf526a2eb18a96cb462abce38de265891`. `multihead_attention`의 head 깊이 스케일과 `dot_product_attention`의 연산 순서를 확인했다.
3. [PyTorch 2.15 — scaled_dot_product_attention](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html), 차원·mask·dropout 및 참조 연산. 사용한 NumPy 실험의 실행 환경과 문서 버전은 구별한다.

확인일: 2026-11-24. 도식·수치 예제는 자체 작성했으며 첨부 코드로 계산을 검증했다.

[이전: SfM과 Bundle Adjustment]({{ '/notes/sfm-bundle-adjustment/' | relative_url }}) · [전체 학습 지도]({{ '/roadmap/' | relative_url }})

**다음 질문:** Attention 결과를 어떻게 쌓고, 토큰의 순서와 위치를 어떻게 알려 줄까? Day 10은 [Transformer와 위치 정보]({{ '/notes/transformer-position/' | relative_url }})다.

