---
layout: post
title: "Detection·Segmentation과 평가: 무엇을 맞혔다고 셀 것인가"
description: "상자·픽셀·개체의 차이에서 시작해 IoU, 중복 매칭, AP와 mIoU를 작은 예제로 검산한다."
date: 2026-11-26 05:00:00 +09:00
category_label: 기초 개념
series_day: 11
topic: "DETECTION & SEGMENTATION"
cover: /assets/day11-iou.svg
tags: [detection, segmentation, evaluation, coco]
reading_time: 10
---

> **1분 요약**
>
> Detection은 물체마다 종류와 상자를 예측한다. Semantic segmentation은 픽셀의 종류를, instance segmentation은 개별 물체의 마스크까지 구분한다. IoU는 두 영역의 겹침 정도다. 검출 평가는 점수 순서대로 예측을 정답에 대응시킨 뒤 precision·recall을 계산한다. 같은 물체를 여러 번 찾았다고 정답 수가 늘어나지 않는다. AP는 순위 전체를 요약하고, mIoU는 클래스별 픽셀 영역의 겹침을 평균한다.
>
> **기억할 문장:** 평가 수치는 예측 표현·매칭 규칙·평균 대상까지 함께 적어야 의미가 있다.

## 1. 전체 내용: 특징 뒤에 필요한 출력과 판정

[Day 10]({{ '/notes/transformer-position/' | relative_url }})은 토큰 표현과 위치 정보를 연결했다. 이제 그 특징으로 무엇을 출력할지 정한다. 같은 이미지라도 사람의 상자, 사람 픽셀 전체, 사람 한 명씩의 마스크는 서로 다른 정답이다. 학습은 정답과 예측의 차이를 줄이지만, 평가에서는 출력들을 비교 가능한 좌표와 형태로 바꾸는 과정도 필요하다. 검출은 개수가 가변이므로 예측과 정답의 대응부터 정해야 한다. 분할은 픽셀 단위 오류를 세되 배경 비중과 클래스 평균 규칙을 확인해야 한다. 이 글은 모델 구조의 우열보다 평가를 읽고 재현하는 기초에 집중한다.

**[원문·공식 코드]**, **[자체 실험]**, **[해석]·[아이디어]**를 구분한다. 실험은 합성 상자와 픽셀 수로 수행한 검산이며 학습된 모델의 성능 보고가 아니다.

## 2. 세 과제는 정답의 단위가 다르다

| 과제 | 이미지 한 장의 대표 출력 | 같은 종류의 두 물체 |
| --- | --- | --- |
| Object detection | 상자 $B\in\mathbb R^{N\times4}$, 클래스 N개, 점수 N개 | 서로 다른 상자로 구분한다 |
| Semantic segmentation | 클래스별 logit $Z\in\mathbb R^{C\times H\times W}$, 최종 label map $H\times W$ | 같은 클래스 픽셀로 묶는다 |
| Instance segmentation | 각 개체의 클래스·점수와 마스크 $N\times H\times W$ | 서로 다른 instance ID로 구분한다 |

N은 예측 개수, C는 클래스 수, H·W는 높이·너비다. 실제 모델은 저해상도 특징이나 ROI 마스크를 출력한 뒤 원영상 크기로 변환하기도 한다. 위 표는 비교를 위해 원영상 격자에 맞춘 형태다. 배치 축은 생략했다.

**[공식 문서]** Torchvision 예제는 semantic logit의 클래스 축 argmax로 픽셀별 label을 만들고, instance 출력은 개별 마스크로 취급한다. [1] Softmax를 적용해도 argmax는 같지만, 이진 마스크의 확률 임계값을 고르는 문제와는 다르다.

**[원문]** COCO 논문 v3 Figure 1은 분류·상자·semantic·instance 표현을 비교한다. 개별 객체를 분리한 마스크는 상자보다 세밀한 위치 표현이다. [2] **[해석]** 두 사람이 붙어 있어도 semantic label은 모두 사람이다. 이를 두 개체로 나누는 능력은 semantic mIoU만으로 측정되지 않는다.

학습에서는 분류 손실과 위치 회귀 손실을 결합하거나 픽셀별 교차 엔트로피를 쓸 수 있다. 그러나 낮은 학습 손실이 특정 평가 지표의 상승과 일대일로 대응하지는 않는다. 후처리·점수 순위·매칭 임계값도 결과를 바꾸기 때문이다.

## 3. IoU: 교집합을 합집합으로 나눈다

예측 영역 A와 정답 영역 G에 대해 다음과 같다. 영역은 상자 내부 또는 이진 마스크의 양성 픽셀 집합이다.

$$
\operatorname{IoU}(A,G)=\frac{|A\cap G|}{|A\cup G|}
=\frac{|A\cap G|}{|A|+|G|-|A\cap G|}
$$

상자는 이 글에서 $(x_1,y_1,x_2,y_2)$ 연속 좌표로 표시한다. 폭은 $x_2-x_1$, 높이는 $y_2-y_1$이며 면적에 +1을 넣지 않는다. COCO 결과의 `bbox`는 $(x,y,w,h)$ 형식이므로 변환이 필요하다. 좌표 규칙을 혼용하면 같은 그림도 IoU가 달라진다.

<figure>
<img src="{{ '/assets/day11-iou.svg' | relative_url }}" alt="가로 좌표 0부터 5, 세로 좌표 0부터 4에서 정답 상자는 0,0,4,4이고 예측 상자는 1,0,5,4다. 교집합 폭 3, 높이 4로 면적 12이며 합집합 면적 20, IoU 0.6이다.">
<figcaption>그림 1. 자체 작성한 정확한 좌표 도식이다. 가로 x, 아래 방향 y이며 단위는 임의 좌표 단위다. 청록색 정답과 노란색 예측의 면적은 각각 16이다. 겹침 12, 합집합 20으로 IoU는 0.6이다. 원문 Figure를 재사용하지 않았다.</figcaption>
</figure>

**[자체 검산]** 그림의 예측은 IoU≥0.5 기준을 통과하지만 IoU≥0.75는 통과하지 못한다. 상자 중심이 가까운 것만으로 충분하지 않으며 크기도 영향을 준다. 작은 물체에서는 같은 한 픽셀 이동이 더 큰 상대 면적 차이를 만들 수 있다.

마스크에는 Dice도 쓰인다.

$$
\operatorname{Dice}(A,G)=\frac{2|A\cap G|}{|A|+|G|}
=\frac{2\operatorname{IoU}}{1+\operatorname{IoU}}
$$

마지막 관계는 같은 비어 있지 않은 두 집합에서 성립한다. 서로 다른 클래스·이미지를 평균한 Dice와 평균 IoU 사이에 그대로 적용할 수는 없다. 둘 다 빈 마스크인 경우 분모가 0이므로 데이터셋의 처리 규칙을 따로 정해야 한다.

## 4. 검출은 먼저 정답과 일대일로 연결한다

**[공식 코드]** COCO의 `evaluateImg`는 이미지·클래스 안에서 점수가 높은 예측부터 매칭한다. 일반 정답은 이미 매칭되었다면 다시 사용하지 않는다. Crowd·ignore에는 별도 규칙이 있다. [3]

아래 설명은 **한 이미지·한 클래스, crowd·ignore 없음**으로 제한한다. 예측마다 아직 사용하지 않은 정답 중 임계값 이상인 가장 큰 IoU를 찾아 연결한다. 연결되면 TP, 연결되지 않으면 FP다. 끝까지 연결되지 않은 정답은 FN이다. 이는 모든 예측과 정답의 전체 IoU 합을 최대로 만드는 최적 할당과는 다르다.

$$
\operatorname{Precision}=\frac{TP}{TP+FP},\qquad
\operatorname{Recall}=\frac{TP}{TP+FN}
$$

상자 위치가 정확해도 클래스를 틀리면 해당 클래스의 TP가 아니다. 정답 한 개를 정확히 복제한 예측 두 개도 일반 매칭에서는 TP 두 개가 되지 않는다.

### 세 임계값을 구분한다

| 설정 | 비교 대상 | 역할 |
| --- | --- | --- |
| Score threshold | 예측의 confidence score | 낮은 점수의 후보를 제거한다 |
| NMS IoU threshold | 예측 상자끼리 | 겹치는 낮은 점수 후보를 억제한다 |
| 평가 IoU threshold | 예측과 정답 | TP로 연결할 위치 정확도를 정한다 |

**[공식 문서]** Torchvision NMS는 높은 점수 상자와 IoU가 기준보다 큰 낮은 점수 상자를 반복 제거한다. [4] NMS는 정답을 보지 않는다. 평가 IoU를 높이는 것과 NMS IoU를 높이는 것은 효과가 다르다. 클래스별 NMS인지도 따로 기록해야 한다. 모든 검출기가 동일한 NMS 절차를 쓰는 것도 아니다.

## 5. AP는 점수 순서 전체를 평가한다

한 confidence threshold에서의 precision은 한 점이다. AP는 점수 내림차순으로 예측을 더하면서 만든 precision–recall 관계를 요약한다. COCO에서는 보간한 precision을 101개 recall 지점에서 평균한다. [3]

$$
p_{\mathrm{interp}}(r)=\max_{\tilde r\ge r}p(\tilde r),\qquad
AP_t=\frac1{101}\sum_{j=0}^{100}p_{\mathrm{interp}}(j/100)
$$

여기서 t는 평가 IoU 임계값이다. 도달하지 못한 recall 지점의 precision은 0으로 둔다. 위 식은 한 클래스·한 IoU의 계산이다. COCO 기본 AP는 IoU 0.50부터 0.95까지 0.05 간격의 10개 조건과 유효 클래스에 걸쳐 평균한다. 기본 전체 면적 AP는 `maxDets=100` 조건을 사용한다. AP50·AP75는 해당 IoU 한 개를 사용한다. 정답이 없는 평가 항목은 코드에서 유효 평균 대상으로 처리되는지 확인해야 한다. [3]

따라서 “mAP 50”이라는 표기만으로 AP50인지, 백분율 50점인지 확정해서는 안 된다. 논문의 정의와 평가 도구를 확인한다. VOC의 다른 AP 관례를 COCO 계산과 섞어서도 안 된다.

### 합성 실험: 중복과 위치 오차를 나눈다

**[자체 실험]** 정답 두 개는 G1=(0,0,4,4), G2=(10,0,14,4)다. 한 이미지·한 클래스, 모두 일반 정답이며 면적 필터·NMS·score 제거는 적용하지 않는다. 예측은 세 개이며 점수 동률은 없다. Python 표준 라이브러리만 사용해 계산했다.

| 순위 | 예측 상자 | 점수 | IoU≥0.50 판정 | 누적 precision / recall |
| --- | --- | --- | --- | --- |
| 1 | (0,0,4,4) | 0.9 | G1과 TP | 1 / 0.5 |
| 2 | (0,0,4,4) | 0.8 | G1 중복 FP | 0.5 / 0.5 |
| 3 | (11,0,15,4) | 0.7 | G2와 TP, IoU=0.6 | 2/3 / 1 |

IoU 0.50에서 보간 precision은 recall 0부터 0.50까지 51개 지점에서 1, 나머지 50개에서 2/3다. 따라서 AP50은 $(51+50\times2/3)/101=0.834983$이다. IoU 0.75에서는 세 번째 예측도 FP가 되어 recall이 0.5에서 멈춘다. AP75는 $51/101=0.504950$이다. 이 두 값은 COCO 전체 데이터셋 점수가 아니라 **합성 예제의 101점 보간 AP**다.

**[해석]** 마지막 precision인 2/3은 AP50과 다르다. 또 이 예제에서 중복을 없애면 AP50은 1이 되지만, 세 번째 상자의 위치는 그대로여서 AP75는 0.504950에 머문다. 중복 억제와 위치 정밀도 개선이 서로 다른 문제임을 보여 준다.

## 6. 픽셀 정확도와 mIoU가 엇갈리는 이유

Semantic segmentation에서는 픽셀별 confusion matrix를 만든다. $M_{ij}$를 실제 클래스 i, 예측 클래스 j인 픽셀 수로 둔다. 클래스 c의 IoU와 평균은 다음과 같다.

$$
IoU_c=\frac{M_{cc}}{\sum_j M_{cj}+\sum_i M_{ic}-M_{cc}},\qquad
mIoU=\frac1{|\mathcal C|}\sum_{c\in\mathcal C}IoU_c
$$

$\mathcal C$는 평가에 포함한 클래스 집합이다. 배경 포함 여부, ignore 픽셀, 분모 0인 클래스의 처리, 이미지별 평균인지 전체 픽셀을 합친 뒤 계산하는지를 명시한다. 큰 클래스에 더 큰 가중치를 주는 pixel accuracy와 클래스별 동일 가중치의 mIoU는 다른 질문에 답한다.

**[자체 실험]** 100픽셀 중 배경 90개, 전경 10개인 이진 정답을 사용한다. 두 클래스 모두 평균에 넣으며 ignore는 없다. 행은 정답, 열은 예측, 순서는 배경·전경이다.

| 예측 | 혼동행렬의 두 행 | Pixel accuracy | 전경 IoU | mIoU | 전경 Dice |
| --- | --- | --- | --- | --- | --- |
| 모두 배경 | (90,0); (10,0) | 0.90 | 0 | 0.45 | 0 |
| 일부 전경 복원 | (85,5); (2,8) | 0.93 | 8/15=0.533333 | 0.728623 | 16/23=0.695652 |

첫 결과의 정확도 90%는 전경을 하나도 찾지 못했다는 사실을 숨긴다. 두 번째 결과의 배경 IoU는 85/92다. 두 클래스의 평균으로 mIoU를 검산했다. 이 예제는 픽셀 수만 다루므로 같은 confusion matrix라도 경계 모양이나 객체가 끊어졌는지는 알 수 없다.

<details markdown="1">
<summary>수치 검산 코드와 적용 범위</summary>

[전체 코드]({{ '/assets/day11-verify.py' | relative_url }})를 `python day11-verify.py`로 실행한다. 연속 좌표 IoU, 일반 일대일 매칭, 101점 보간 AP, 픽셀 지표를 assertion으로 확인한다. 난수·학습·외부 패키지는 없다. 공식 COCO evaluator 전체를 실행한 결과가 아니며 crowd·ignore·면적 범위·여러 이미지 집계를 생략했다. 실제 연구 평가는 공식 도구와 버전을 고정한다.

</details>

## 7. Figure를 읽는 관점과 한계

**[원문]** COCO v3 Figure 9, PDF 10쪽은 상자 위치와 마스크 품질을 분리해 보여 준다. 가운데 산점도는 가로축 bounding-box overlap, 세로축 segmentation overlap이며 person 사례다. §6.2의 실험은 상자 IoU≥0.5인 올바른 검출에 조건을 걸어 마스크 겹침을 평가한다. [2] 이는 현재 공식 코드의 mask AP와 같은 평가가 아니다.

**[해석]** 상자가 맞았다고 객체 경계가 정확하다는 보장은 없다. 반대로 semantic mIoU가 높아도 가까운 같은 클래스 개체를 구분했는지는 알 수 없다. 지표를 선택할 때 응용에서 필요한 오류 단위를 먼저 정해야 한다.

COCO 부록 I는 개별 instance 표시·분할·검증과 crowd labeling을 구분한다. [2] 실제 평가에서 crowd를 일반 개체 여러 개로 취급하는 단순 예제를 그대로 적용하면 안 된다. 주석 정책은 평가 정의의 일부다.

| 해석의 위험 | 함께 확인할 정보 |
| --- | --- |
| AP 상승을 전부 위치 개선으로 해석한다 | AP50·AP75, 클래스 오류, 점수 순위, 중복 |
| 서로 다른 데이터셋의 mIoU를 직접 비교한다 | 클래스 수, 배경·ignore 처리, 주석 정책 |
| 높은 AP를 신뢰할 확률로 읽는다 | confidence calibration은 별도 평가다 |
| 평균 점수로 모든 실패를 설명한다 | 작은 물체, 가림, 얇은 경계의 사례와 부분집합 |
| 시간별 안정성까지 평가했다고 본다 | 단일 프레임 지표는 시간적 깜박임을 직접 측정하지 않는다 |

## 8. 검증 가능한 확장 아이디어

**[아이디어 · 미실행]** 중복 억제 효과와 위치 개선을 분리하는 작은 평가 실험이다.

- **가설:** 겹친 물체가 많은 장면에서 강한 NMS는 중복 FP를 줄이지만 서로 다른 실제 물체도 제거해 recall을 낮출 수 있다.
- **Baseline:** 동일 검출기 체크포인트의 기본 후처리다. NMS 직전 후보를 저장해 다시 사용한다.
- **최소 실험:** 고정한 검증 이미지 500장에서 클래스별 NMS IoU를 0.3·0.5·0.7로 바꾼다. 이미지 목록과 선정 규칙을 결과 확인 전에 고정한다.
- **통제:** 가중치·입력 해상도·전처리·score cutoff·최대 후보 수·평가 코드를 고정한다. NMS를 원래 쓰는 모델에 한정하며 학습은 하지 않는다.
- **지표:** AP50·AP75·AR100, 중복 FP 수, 남은 예측 수를 기록한다. 같은 클래스 정답 상자끼리 IoU>0.1인 쌍을 포함한 이미지 집합을 사전에 정의해 별도로 비교한다.
- **기각 기준:** IoU threshold를 낮춰도 중복 감소와 해당 집합 recall 감소가 나타나지 않으면 가설을 지지하지 않는다. 고정 이미지 단위 paired bootstrap 1,000회로 차이의 불확실성도 보고한다.

실행 전 제안이다. 가장 좋은 threshold를 평가 데이터에서 고른 뒤 같은 데이터 점수를 최종 일반화 성능으로 보고하지 않는다. 설정 선택용 데이터와 최종 평가 데이터를 분리한다.

## 9. 복습

1. 같은 정답을 정확히 복제한 두 검출이 왜 TP 두 개가 아닌가?
2. AP50과 confidence threshold 0.5는 같은 조건인가?
3. 배경이 90%인 영상에서 픽셀 정확도 90%만으로 전경 분할이 성공했다고 말할 수 있는가?

<details markdown="1">
<summary>답</summary>

1. 일반 정답에 대한 일대일 매칭에서 먼저 매칭한 예측만 TP이고 나머지는 중복 FP이기 때문이다. Crowd 규칙은 별도다.
2. 아니다. AP50의 0.5는 예측–정답 IoU 기준이며 confidence는 후보의 점수다.
3. 아니다. 모두 배경으로 예측해도 90%다. 전경 IoU·클래스별 지표와 실패 사례를 함께 확인해야 한다.

</details>

## 출처와 다음 읽기

1. [Torchvision 공식 시각화 예제](https://docs.pytorch.org/vision/stable/auto_examples/others/plot_visualization_utils.html): 상자 좌표와 semantic·instance 출력 절을 확인했다.
2. Lin et al., **Microsoft COCO: Common Objects in Context**, [arXiv:1405.0312v3, 2015-02-21](https://arxiv.org/pdf/1405.0312v3). Figure 1·9, §6.2, Appendix I의 주석 구분을 확인했다. 원 연구는 ECCV 2014이며 이 글은 확장된 v3를 참조한다.
3. [COCO 공식 cocoeval.py](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py): `evaluateImg`, `accumulate`, `summarize`, `Params.setDetParams`를 확인했다. 현재 평가 설명은 이 코드에 근거하며 초기 논문 실험과 구분했다.
4. [Torchvision 공식 NMS 문서](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.nms.html): 제거 기준과 좌표 형식을 확인했다.

확인일: 2026-11-26. 자체 도식·합성 검산이며 실제 모델 학습이나 COCO benchmark 재현은 수행하지 않았다.

[이전: Transformer와 위치 정보]({{ '/notes/transformer-position/' | relative_url }}) · [전체 학습 지도]({{ '/roadmap/' | relative_url }})

**다음 질문:** 정답 라벨이 적을 때도 좋은 특징을 얻을 수 있을까? 다음 글은 [Day 12: 표현 학습·대조 학습]({{ '/notes/representation-contrastive-learning/' | relative_url }})이다.

