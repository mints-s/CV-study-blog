---
layout: post
title: "Transformer와 위치 정보: 무엇을 섞고 어디에 놓는가"
description: "Attention·FFN·잔차·LayerNorm을 한 블록으로 연결하고, 위치 부호가 토큰 순서의 의미를 바꾸는 과정을 검산한다."
date: 2026-11-25 04:50:00 +09:00
category_label: 기초 개념
series_day: 10
topic: "TRANSFORMER & POSITION"
cover: /assets/day10-block.svg
tags: [transformer, positional-encoding, layernorm, vision]
reading_time: 10
---

> **1분 요약**
>
> Transformer 블록은 Attention만으로 이루어지지 않는다. Attention은 토큰 사이 정보를 모으고, FFN은 각 토큰의 채널 표현을 바꾼다. 잔차 연결은 기존 표현을 더하는 경로이며 LayerNorm은 지정한 특징 축을 정규화한다. 위치 정보가 없는 블록은 토큰의 내용과 순서를 함께 재배열하면 출력도 같은 순서로 바뀐다. 위치 부호는 내용과 위치의 대응을 알려 주는 한 방법이다.
>
> **기억할 문장:** 토큰 배열의 저장 순서와 모델에 전달된 위치 정보는 다르다. 무엇을 재배열하고 무엇을 고정했는지를 먼저 정해야 한다.

## 1. 전체 내용: Attention을 반복 가능한 블록으로 만든다

[Day 9]({{ '/notes/attention-qkv/' | relative_url }})에서는 Q·K로 가중치를 만들고 V를 섞었다. 이 연산을 깊게 쌓으려면 입력과 출력의 차원을 맞추고, 특징의 크기를 조절하며, 토큰별 비선형 변환도 넣어야 한다. Transformer는 이런 요소를 블록으로 결합한다. 한 블록이 다른 토큰의 정보를 모으면 다음 블록은 갱신된 표현을 다시 읽는다. 여기에 위치 정보가 더해지면 같은 내용이 어느 자리에 있는지도 계산에 영향을 준다. 이미지에서는 패치의 좌표, 영상에서는 공간과 시간의 구분이 중요하다.

이 글은 encoder 블록을 중심으로 한 기초 설명이다. 원래의 번역 Transformer와 ViT의 정규화 순서를 구분한다. **[원문·공식 코드]**, **[자체 검산]**, **[해석]·[아이디어]**를 나누며, 아래 수치 실험은 학습 성능 비교가 아니다.

## 2. 네 구성 요소의 역할

배치 축을 생략한 입력은 $X\in\mathbb R^{n\times d}$다. n은 토큰 수, d는 특징 차원이다.

| 구성 | 하는 일 | 하지 않는 일 |
| --- | --- | --- |
| Multi-head attention | 다른 토큰의 내용을 조건부 가중합으로 모은다 | 올바른 물리적 대응을 자동 보장하지 않는다 |
| FFN/MLP | 각 토큰에 같은 채널 변환을 독립 적용한다 | 표준 position-wise FFN 자체는 토큰끼리 섞지 않는다 |
| Residual addition | 같은 차원의 입력과 변환 결과를 더한다 | 정규화나 차원 변환을 대신하지 않는다 |
| LayerNorm | 지정한 특징 축의 평균·분산으로 정규화한다 | 기본 토큰별 설정에서는 토큰 순서나 공간 좌표를 만들지 않는다 |

**[원문]** 2017 Transformer의 encoder는 attention과 position-wise FFN을 잔차·정규화와 결합한다. FFN은 두 선형 변환 사이에 ReLU를 둔다. [1] 한 토큰의 행벡터 x에 대해 다음과 같다.

$$
\operatorname{FFN}(x)=\phi(xW_1+b_1)W_2+b_2
$$

$W_1\in\mathbb R^{d\times d_{ff}}$, $W_2\in\mathbb R^{d_{ff}\times d}$다. 토큰 수 n은 유지하고 내부 채널 폭을 바꾼다. 같은 층에서는 토큰들이 W를 공유하지만, 서로 다른 층의 W까지 공유해야 하는 것은 아니다. ViT의 확인한 공식 구현은 GELU를 사용한다. [2, 3]

## 3. Pre-LN과 Post-LN은 순서가 다르다

dropout을 생략하면 다음처럼 비교할 수 있다. MSA는 입력·출력 차원이 d인 multi-head self-attention이다.

| 방식 | Attention 부분 | FFN 부분 |
| --- | --- | --- |
| Post-LN | $U=\mathrm{LN}(X+\mathrm{MSA}(X))$ | $Y=\mathrm{LN}(U+\mathrm{FFN}(U))$ |
| Pre-LN | $U=X+\mathrm{MSA}(\mathrm{LN}(X))$ | $Y=U+\mathrm{FFN}(\mathrm{LN}(U))$ |

**[원문·공식 코드]** 원래 Transformer는 post-LN 형태이고, ViT 본문 식 (2)–(3) 및 `Encoder1DBlock`은 pre-LN 형태다. ViT 구현에는 encoder 스택 뒤의 마지막 LN도 있다. [1–3] 블록 내부 순서와 마지막 LN의 존재를 따로 확인해야 한다.

<figure>
<img src="{{ '/assets/day10-block.svg' | relative_url }}" alt="Pre-LN 블록에서 X는 LN과 Attention을 거친 값에 더해져 U가 된다. U는 LN과 FFN을 거친 값에 더해져 Y가 된다. 두 잔차 경로는 LN을 우회한다.">
<figcaption>그림 1. 자체 작성한 pre-LN encoder 블록이다. 모든 주요 경로의 출력은 n×d다. 두 우회 화살표가 잔차 경로이며 dropout과 스택 뒤 최종 LN은 생략했다. 원문 Figure의 재사용이 아니다.</figcaption>
</figure>

**[해석]** 두 식을 같은 모델이라고 보고 가중치를 그대로 옮기면 연산이 달라진다. “어느 방식이 항상 더 좋다”는 결론은 이 구조 비교만으로 얻지 못한다. 깊이·초기화·학습률 등 학습 조건을 통제한 별도 실험이 필요하다.

### LayerNorm의 축을 확인한다

토큰 한 개의 d차원 특징 $x$에 대해 다음과 같이 정의한다.

$$
\mu=\frac1d\sum_{k=1}^d x_k,\quad
\sigma^2=\frac1d\sum_{k=1}^d(x_k-\mu)^2,\quad
\mathrm{LN}(x)_k=\gamma_k\frac{x_k-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_k
$$

**[공식 문서]** PyTorch는 `normalized_shape`가 지정한 마지막 축들을 사용하고, 분산의 분모는 원소 수다. 학습·평가 모두 입력에서 통계를 계산한다. [4] 일반적인 `(batch,n,d)` 토큰 텐서에 `LayerNorm(d)`를 쓰면 토큰별 d축을 정규화한다. `LayerNorm((n,d))`는 다른 연산이다.

**[자체 검산]** $x=(1,2,3,4)$, $\gamma=1$, $\beta=0$, $\epsilon=10^{-5}$이면 평균은 2.5, 분산은 1.25다. 출력은 약 $(-1.341635,-0.447212,0.447212,1.341635)$다. 출력 분산은 약 0.999992이며 epsilon 때문에 정확히 1은 아니다. 학습 가능한 gamma·beta까지 적용하면 평균 0·분산 1도 일반적 보장이 아니다.

## 4. 위치가 없으면 무엇을 구별하지 못하는가

각 토큰에 같은 FFN과 LN을 적용하고, 위치 bias·mask·dropout이 없다고 하자. 토큰 순열 행렬을 $P\in\mathbb R^{n\times n}$라 하면 블록 f는 다음을 만족한다.

$$
f(PX)=Pf(X)
$$

**[해석·자체 검산]** 순서를 바꾸면 출력도 따라 바뀌는 순열 등변성이다. “출력이 모두 같다”는 뜻은 아니다. 평균 pooling까지 하면 재배열한 결과의 평균은 같으므로 그 readout은 순열 불변이 된다. 이미지의 위치를 제거한 채 패치 집합의 평균으로 분류한다면 패치 내용의 배열 차이를 구별하기 어렵다.

이 조건에 맞지 않는 경우도 있다. 입력 특징에 좌표가 이미 포함되거나 공간 구조를 활용하는 전처리가 있다면, 단순히 positional embedding이 없다는 이유로 모든 위치 정보가 사라졌다고 단정할 수 없다. Causal mask도 허용 연결에 순서를 반영한다.

## 5. 위치 부호를 더한다는 뜻

내용 임베딩 X와 위치 행렬 $E\in\mathbb R^{n\times d}$를 더해 $Z_0=X+E$로 입력할 수 있다. 더하기이므로 채널 차원 d는 늘어나지 않는다. 모델은 내용과 위치가 결합된 표현으로 Q·K·V를 계산한다.

**[원문]** 원래 Transformer는 다음 sin/cos 부호를 사용했다. p는 0부터 시작하는 위치, $i=0,\ldots,d/2-1$이며 d는 짝수로 둔다. [1]

$$
E_{p,2i}=\sin(p\omega_i),\qquad
E_{p,2i+1}=\cos(p\omega_i),\qquad
\omega_i=10000^{-2i/d}
$$

**[자체 계산]** d=4이면 주파수는 1과 0.01이다.

| p | sin(p) | cos(p) | sin(0.01p) | cos(0.01p) |
| --- | --- | --- | --- | --- |
| 0 | 0.000000 | 1.000000 | 0.000000 | 1.000000 |
| 1 | 0.841471 | 0.540302 | 0.010000 | 0.999950 |
| 2 | 0.909297 | -0.416147 | 0.019999 | 0.999800 |

위치 0의 벡터가 0 벡터가 아니라는 점에 주의한다. 또한 느리게 변하는 채널 하나만 보면 가까운 위치들이 비슷하지만 전체 채널 묶음은 다르다.

<details markdown="1">
<summary>위치 부호의 내적이 상대 차이에 연결되는 이유</summary>

삼각함수 항등식으로 한 sin/cos 쌍의 내적은 다음과 같다.

$$
\sin(p\omega)\sin(q\omega)+\cos(p\omega)\cos(q\omega)
=\cos((p-q)\omega)
$$

d=4 예제에서 $E_1^\top E_2=\cos(1)+\cos(0.01)\approx1.540252$다. 직접 계산해 확인했다. 하지만 실제 attention 점수는 내용과 학습 투영을 포함한다. $X+E$의 QKᵀ에는 내용–내용, 내용–위치, 위치–내용, 위치–위치 항이 함께 들어간다. 따라서 모델의 점수가 오직 상대 거리로만 결정된다는 결론은 성립하지 않는다.

</details>

Sin/cos 부호는 학습 범위 밖 위치에도 계산할 수 있지만, 계산 가능성과 학습하지 않은 길이에서의 성능은 다르다. 학습형 위치 표도 유용하며 서로 다른 방법의 우열은 과제와 조건에 따라 확인해야 한다.

## 6. 수치 실험: 내용 이동과 단순 재인덱싱은 다르다

**[자체 실험]** 다음 3개 토큰과 위 E를 사용했다.

$$
X=\begin{bmatrix}1&0&0&1\\0&1&1&0\\1&1&0&0\end{bmatrix}
$$

비교하는 순열은 원래 0·1·2번 행을 **2·0·1번 순서**로 옮기는 것이다. f는 한 층 pre-LN 블록이다. 조건은 NumPy 2.3.5, float64, 단일 head, d=4, Q/K/V/출력 투영 모두 항등, attention scale=1/2다. FFN은 $W_1=W_2=I$, bias=0, $d_{ff}=4$, ReLU다. LN은 gamma=1, beta=0, epsilon=1e-5다. Dropout·mask·마지막 LN·학습·난수는 없다. **ViT 구조를 그대로 재현한 실험이 아니다.**

| 비교 | 의미 | 최대 원소 절댓값 차이 |
| --- | --- | --- |
| $f(PX)$와 $Pf(X)$ | 위치 없이 내용 재배열 | $10^{-12}$ 미만 |
| $f(P(X+E))$와 $Pf(X+E)$ | 내용과 위치를 함께 재인덱싱 | $10^{-12}$ 미만 |
| $f(PX+E)$와 $Pf(X+E)$ | 내용만 다른 고정 위치로 이동 | 5.471265 |

기준 $f(X+E)$의 첫 행은 약 $(1.470864,1.069657,-1.277642,4.248618)$이다. 두 번째 조건은 내용과 위치의 짝을 보존하므로 단순 저장 순서 변경이다. 세 번째 조건은 같은 내용을 다른 위치 벡터와 결합하므로 입력 관계 자체가 바뀐다.

**범위:** 5.471265는 이 고정 파라미터에서의 특징값 차이다. 정확도나 위치 인식 능력의 점수가 아니다. 학습된 모델이 위치를 충분히 활용한다거나 모든 내용 이동을 구별한다는 보장도 아니다.

<details markdown="1">
<summary>재현 코드</summary>

[전체 검증 코드]({{ '/assets/day10-verify.py' | relative_url }})를 NumPy 2.3.5에서 `python day10-verify.py`로 실행한다. 위치 부호, LN, 세 가지 재배열, 위치 내적을 assertion으로 확인한다. 핵심 블록은 다음과 같다.

~~~python
z = ln(x)
s = z @ z.T / 2
weights = np.exp(s - s.max(-1, keepdims=True))
weights /= weights.sum(-1, keepdims=True)
u = x + weights @ z
y = u + np.maximum(ln(u), 0)
~~~

첫 잔차는 x, 두 번째 잔차는 u를 그대로 더한다. FFN을 단순화해 ReLU만 남긴 것은 검산 조건의 항등 가중치 때문이다. 일반 FFN이 항상 ReLU 하나라는 뜻은 아니다.

</details>

## 7. 이미지의 2D 위치와 해상도

**[원문·공식 코드]** ViT는 패치 임베딩에 학습형 위치 부호를 더한다. 확인한 구현의 `AddPositionEmbs`는 `(1,seq_len,emb_dim)` 파라미터를 입력에 더한다. [2, 3] 1D 표라도 각 index를 고정된 2D 패치 위치에 대응시킬 수 있다. “1D 위치 표”와 “공간 배열을 전혀 모른다”는 같은 말이 아니다.

자체 차원 예로 224×224 RGB 영상을 16×16 패치로 나누면 14×14=196개 패치이며, 각 패치를 펼친 길이는 16×16×3=768이다. 이를 d차원으로 투영하고 class token 하나를 더하면 시퀀스는 197×d다. 384×384에서는 패치가 576개, class token을 포함하면 577개가 된다. 고정 크기의 위치 표를 그대로 더할 수 없으므로 격자 크기와 보간 등의 처리 규칙을 확인해야 한다. Class token 위치와 패치 격자를 섞어 취급해서도 안 된다.

**[원문]** ViT 부록 D.4·Table 8은 ViT-B/16의 ImageNet 5-shot linear 평가에서 위치 부호 유형과 삽입 위치를 비교한다. 기본 입력 삽입 조건의 1D와 2D 부호 결과는 각각 0.64206, 0.64001이다. [2] **[해석]** 이 한 조건을 “2D 부호가 언제나 불필요하다”는 근거로 쓰면 안 된다. 패치 크기·해상도·훈련·평가 과제가 달라지면 별도 검증이 필요하다. 본 글에서는 이 결과를 재현하지 않았다.

## 8. 한계와 검증 가능한 확장 아이디어

| 오개념 | 점검할 조건 |
| --- | --- |
| Attention을 쌓으면 좌표를 자동으로 안다 | 입력 위치 정보, mask, 공간 전처리 여부 |
| LN은 batch 전체를 같은 방식으로 평균낸다 | normalized_shape와 실제 텐서 축 |
| 위치 부호는 항상 더하기다 | 상대 위치 bias나 다른 주입 방식의 존재 |
| 고해상도에서는 위치 표만 늘리면 끝난다 | 패치 수, 계산량, 보간, 학습 분포 변화 |
| Decoder는 encoder와 완전히 같다 | causal self-attention, cross-attention 등 구조 차이 |

**[아이디어 · 미검증]** 해상도가 달라질 때 위치 표 처리의 효과와 패치 수 증가의 효과를 분리해 본다.

- **가설:** 고정 해상도에서 배운 절대 위치 표의 처리 방식이 해상도 변경 시 일부 오차를 설명한다.
- **비교:** 동일 체크포인트에 정상적인 2D 격자 보간, 위치 벡터 0, 패치 위치 표의 고정 무작위 재배열을 각각 적용한다. Class token 위치는 보존한다.
- **최소 실험:** 학습에 사용하지 않은 동일 이미지 1,000개를 224·384 해상도로 평가한다. 패치 크기·crop 영역·정규화·가중치를 고정하며 재학습하지 않는다.
- **지표:** 분류 정확도, 같은 이미지의 예측 일치율, 출력 특징 변화량, 최고 메모리를 기록한다.
- **통제:** 같은 해상도 안에서 위치 처리 방식끼리 비교해 토큰 수를 맞춘다. 위치 재배열은 5개 seed로 반복한다.
- **기각 기준:** 같은 해상도에서 처리 방식의 차이가 일관되지 않으면 위치 표 설명을 지지하지 않는다. 224와 384의 차이 전체를 위치 부호 탓으로 돌리지 않는다.

실행하지 않은 제안이다. 앞의 고정 블록 검산은 이 분류 실험의 결과가 아니다. 또한 사후 위치 제거는 학습하지 않은 입력 변화이므로 “위치 없이 처음부터 훈련한 모델”의 성능을 대신하지 않는다.

## 9. 복습

1. Attention과 position-wise FFN 중 어느 연산이 토큰 사이 정보를 직접 섞는가?
2. Pre-LN에서 잔차 경로도 반드시 LN을 통과하는가?
3. $PX+E$와 $P(X+E)$는 왜 다른 실험인가?

<details markdown="1">
<summary>답</summary>

1. Attention이다. 표준 FFN은 각 토큰에 같은 채널 변환을 독립 적용한다.
2. 아니다. 이 글의 pre-LN 블록에서 잔차 경로는 LN을 우회해 입력을 더한다. 스택 뒤 최종 LN은 별도다.
3. 전자는 위치를 고정하고 내용만 이동한다. 후자는 내용과 위치를 함께 재인덱싱해 원래의 짝을 보존한다.

</details>

## 출처와 다음 읽기

1. Vaswani et al., **Attention Is All You Need**, NIPS 2017, [arXiv v7](https://arxiv.org/html/1706.03762v7). §3.1·3.3·3.5의 post-LN, FFN, sinusoidal encoding을 확인했다.
2. Dosovitskiy et al., **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale**, ICLR 2021, [arXiv v2](https://arxiv.org/html/2010.11929v2). §3.1–3.2의 구조와 고해상도 처리, 부록 D.4·Table 8을 확인했다. 정규 ViT 논문 리뷰는 Day 18에서 다룰 예정이다.
3. [공식 ViT models_vit.py](https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py), 확인 blob `6a384032751671eff811da2b5e8ca3e5622dfe09`. AddPositionEmbs·Encoder1DBlock·Encoder·MlpBlock을 확인했다.
4. [PyTorch 2.15 LayerNorm](https://docs.pytorch.org/docs/2.15/generated/torch.nn.LayerNorm.html), 정규화 축·분산·affine·epsilon 정의.

확인일: 2026-11-25. 자체 도식과 수치 검산을 사용했으며 원문 Figure를 재사용하지 않았다. 공식 JAX·PyTorch 모델을 실행한 결과가 아니다.

[이전: Attention의 Q, K, V]({{ '/notes/attention-qkv/' | relative_url }}) · [전체 학습 지도]({{ '/roadmap/' | relative_url }})

**다음 질문:** 특징을 얻은 뒤 물체의 위치와 픽셀별 영역을 어떻게 예측하고 평가할까? 다음 글은 [Day 11: Detection·Segmentation과 평가]({{ '/notes/detection-segmentation-evaluation/' | relative_url }})다.

