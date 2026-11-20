---
layout: post
title: "Epipolar Geometry: 대응점은 왜 한 직선 위에 놓이는가"
description: "한 픽셀의 광선을 다른 카메라로 바라보면 직선이 된다. E·F와 대응점 검증을 평행 스테레오 예제로 연결한다."
date: 2026-11-21 05:00:00 +09:00
category_label: "기초 개념"
series_day: 6
topic: "EPIPOLAR GEOMETRY"
cover: /assets/day06-epipolar.svg
tags: [epipolar, essential-matrix, fundamental-matrix, stereo]
reading_time: 10
---

> **1분 요약**
>
> 한 영상의 픽셀은 3D 점 하나가 아니라 광선을 정한다. 그 광선 위의 가능한 점들을 다른 카메라에 투영하면 일반적으로 **epipolar line**이 된다. Essential matrix E는 정규화 좌표, fundamental matrix F는 픽셀 좌표의 관계를 표현한다. 대응점은 $p_2^\top Fp_1=0$을 만족해야 하지만, 이 조건만으로 올바른 대응이라고 확정할 수는 없다.
>
> **기억할 문장:** F는 점을 점으로 옮기지 않는다. 점이 놓일 직선을 정한다.

## 1. 전체 내용: 두 광선이 만날 수 있는 조건

[Day 4]({{ '/notes/pinhole-camera-intrinsics-extrinsics/' | relative_url }})에서는 픽셀의 역투영이 광선임을 확인했고, [Day 5]({{ '/notes/camera-calibration/' | relative_url }})에서는 투영 모델을 추정했다. 이제 두 카메라가 같은 장면을 본다고 하자. 첫 픽셀만으로 깊이를 정할 수 없지만, 다른 카메라에서 대응점이 있을 범위는 제한할 수 있다. 두 카메라 중심과 첫 광선이 만드는 평면이 두 번째 영상 평면과 만나는 선이 그 범위다. 기하 제약으로 후보를 줄이고, 외관이나 추가 관측으로 그중 대응점을 선택한다. 대응이 정해진 뒤 3D 점을 추정하는 단계가 다음 회차의 triangulation이다.

이 글은 정적인 장면, 왜곡 없는 pinhole 좌표, 서로 다른 카메라 중심을 기본 조건으로 둔다. 개념과 점-직선 관계는 OpenCV 공식 자료 [1, 2]를 확인했다. 아래 유도·숫자·도식은 자체 예제이며 대응점 검출이나 실제 영상의 자세 추정 결과가 아니다.

## 2. 평면·직선·epipole을 구별한다

첫 카메라 중심을 $C_1$, 두 번째를 $C_2$라 하자. $C_1,C_2$와 3D 점 P를 포함하는 평면이 **epipolar plane**이다. 그 평면이 각 영상 평면과 만나는 선이 epipolar line이다. 두 카메라 중심을 잇는 baseline이 영상 평면과 만나는 점은 epipole이며, 다른 카메라 중심의 투영으로도 해석한다. [1]

첫 영상의 점을 고정하고 깊이만 바꾸면 P는 첫 광선을 따라 움직인다. 카메라 중심과 그 광선이 만드는 평면은 그대로이므로 두 번째 투영점은 같은 직선 위에서 움직인다. 단, 광선이 baseline 자체와 겹치는 특수한 경우에는 이 설명의 평면이 유일하지 않다.

Epipole은 화면 밖이나 무한원에 있을 수 있다. 평행 스테레오에서 epipolar line이 서로 평행한 것은 무한원 epipole로 이해할 수 있다. **일반적인 두 이미지의 epipolar line이 항상 수평인 것은 아니다.** Rectification은 대응 epipolar line을 정렬하기 위한 영상 좌표 변환이다.

## 3. E의 유도: 같은 평면이라는 조건을 내적으로 쓴다

열벡터를 사용하고, 카메라 1 좌표에서 카메라 2 좌표로의 변환을 다음처럼 고정한다.

$$
\mathbf P_2=R\mathbf P_1+t
$$

왜곡 없는 동차 픽셀을 $p_i=(u_i,v_i,1)^\top$, 정규화 광선 벡터를 $x_i=K_i^{-1}p_i$라 둔다. $x_i$의 마지막 성분은 1이며 단위 길이 벡터일 필요는 없다. 양의 깊이 $\lambda_i$에 대해 $\mathbf P_i=\lambda_i x_i$이므로 다음 관계가 성립한다.

$$
\lambda_2x_2=\lambda_1Rx_1+t
$$

$t$와 $Rx_1$이 만드는 평면의 법선은 $t\times Rx_1$이다. $x_2$도 그 평면에 있으므로 내적이 0이다.

$$
x_2^\top[t]_\times Rx_1=0,\qquad E=[t]_\times R
$$

$$
[t]_\times=
\begin{bmatrix}
0&-t_z&t_y\\
t_z&0&-t_x\\
-t_y&t_x&0
\end{bmatrix}
$$

$[t]_\times v=t\times v$다. E는 $3\times3$이며 $t\ne0$인 이상적 경우 rank 2다. 상대 자세의 방향을 뒤집으면 행렬도 달라진다. 이 글의 t는 카메라 2 중심의 카메라 1 좌표가 아니라, 위 변환식의 이동 항이다.

<details markdown="1">
<summary>세계 좌표계의 두 자세에서 상대 자세 구하기</summary>

두 카메라가 $\mathbf P_1=R_1\mathbf P_w+t_1$, $\mathbf P_2=R_2\mathbf P_w+t_2$를 사용하면 다음과 같다.

$$
R=R_2R_1^\top,\qquad t=t_2-R_2R_1^\top t_1
$$

첫 식을 역변환해 두 번째 식에 대입하면 얻는다. 한 데이터셋의 world-to-camera 행렬과 다른 코드의 camera-to-world 행렬을 혼용하지 않는 것이 출발점이다.

</details>

## 4. F는 픽셀을 직선으로 보낸다

$x_i=K_i^{-1}p_i$를 E의 제약에 대입하면 다음과 같다.

$$
p_2^\top Fp_1=0,\qquad F=K_2^{-\top}EK_1^{-1}
$$

| 대상 | 입력 좌표 | 제약·출력 |
| --- | --- | --- |
| E | 왜곡 없는 정규화 좌표 $x_i$ | $x_2^\top Ex_1=0$ |
| F | 왜곡 없는 픽셀 좌표 $p_i$ | $p_2^\top Fp_1=0$ |
| $l_2=Fp_1$ | 첫 영상의 점 | 두 번째 영상의 직선 |
| $l_1=F^\top p_2$ | 두 번째 영상의 점 | 첫 영상의 직선 |

$l_2=(a,b,c)^\top$라면 두 번째 영상의 후보 $(u,v)$는 $au+bv+c=0$을 만족한다. OpenCV의 computeCorrespondEpilines 문서도 이 방향을 사용한다. [2] F와 0이 아닌 상수배 F는 같은 제약을 나타낸다.

**[해석]** $Fp_1$을 마지막 성분으로 나누어 “예측 픽셀”로 읽으면 잘못된 연산이다. 여기서는 출력이 점이 아니라 직선 계수다. 또한 왜곡된 원본 픽셀을 그대로 사용하면 이상적인 직선 제약과 어긋날 수 있다. [Day 5]({{ '/notes/camera-calibration/' | relative_url }})의 보정 좌표와 K를 일관되게 사용한다.

## 5. 숫자와 그림: 수평 baseline의 예제

두 카메라의 축이 같고 카메라 2가 카메라 1의 오른쪽으로 $B=0.2$ m 떨어져 있다고 하자. 두 내부 행렬은 동일하고 $f_x=f_y=800$, $(c_x,c_y)=(320,240)$이다. 영상 크기는 $640\times480$이다.

카메라 1을 세계 좌표로 두면 $R=I$, $t=(-0.2,0,0)^\top$다. 점 $(0.5,0.25,2)$ m의 카메라 2 좌표는 $(0.3,0.25,2)$다. 따라서 다음을 얻는다.

$$
p_1=(520,340,1)^\top,\qquad p_2=(440,340,1)^\top
$$

$$
E=
\begin{bmatrix}0&0&0\\0&0&0.2\\0&-0.2&0\end{bmatrix},
\qquad
F\sim
\begin{bmatrix}0&0&0\\0&0&1\\0&-1&0\end{bmatrix}
$$

$\sim$는 0이 아닌 상수배까지 같다는 뜻이다. 실제 $K^{-\top}EK^{-1}$는 위 표시 행렬의 $0.00025$배다. 표시한 F로 계산하면 $l_2=(0,1,-340)^\top$, 즉 **v=340**이다.

<figure>
<img src="{{ '/assets/day06-epipolar.svg' | relative_url }}" alt="첫 영상의 (520,340)은 두 번째 영상의 수평선 v=340을 정한다. 깊이 2m의 대응점은 (440,340), 같은 첫 광선에서 깊이 4m인 다른 점의 투영은 (480,340)이다.">
<figcaption>그림 1. 직접 계산한 평행 스테레오 예제. 두 영상은 같은 축척으로 표시했다. 오른쪽의 두 점은 첫 광선 위 서로 다른 깊이 가설이며 동시에 같은 물체의 대응점이라는 뜻이 아니다.</figcaption>
</figure>

첫 광선 위에서 깊이를 4 m로 바꾼 세계 점은 $(1,0.5,4)$다. 첫 픽셀은 그대로지만 두 번째 픽셀은 $(480,340)$이 된다. 깊이가 달라도 같은 epipolar line을 만족한다.

| 깊이 Z | 첫 영상 u₁ | 두 번째 u₂ | disparity $d=u_1-u_2$ |
| --- | --- | --- | --- |
| 2 m | 520 | 440 | 80 pixel |
| 4 m | 520 | 480 | 40 pixel |

이 **동일 K·평행 축·수평 baseline** 조건에서 $d=f_xB/Z$다. 두 번째 점을 고르면 깊이도 달라지므로 직선 제약만으로 정답을 결정하지 못한다. 이 예제의 disparity 공식을 임의의 비정렬 영상에 그대로 적용하면 안 된다.

## 6. 잔차가 0이면 정말 정답인가?

아니다. 같은 선 위의 잘못된 외관 대응도 제약을 통과한다. 예제에서 $(400,340)$은 실제 주어진 점의 대응 $(440,340)$과 다르지만 $p_2^\top Fp_1=0$이다.

또한 대수 잔차 $p_2^\top Fp_1$ 자체는 픽셀 거리가 아니다. F의 배율에 따라 값이 변하기 때문이다. $l=(a,b,c)$가 유효한 직선이고 $a^2+b^2>0$이면 한쪽 점-직선 거리는 다음과 같다.

$$
d_{\mathrm{line}}=\frac{|au+bv+c|}{\sqrt{a^2+b^2}}
$$

후보를 $(440,343)$으로 바꾸면 표시한 F의 대수 잔차는 3, $10F$에서는 30이다. 하지만 직선 거리는 둘 다 **3 pixel**이다. 이 거리는 두 영상의 관측을 함께 보정하는 최적 재투영 오차와는 다른 지표다.

<details markdown="1">
<summary>실행한 투영·E/F·거리 검증 코드</summary>

~~~python
from math import isclose, hypot

def mv(A, x):
    return [sum(a*b for a,b in zip(row,x)) for row in A]
def dot(a, b):
    return sum(x*y for x,y in zip(a,b))
def transpose(A):
    return [list(row) for row in zip(*A)]
def mm(A, B):
    return [[dot(row,col) for col in transpose(B)] for row in A]

E = [[0,0,0], [0,0,0.2], [0,-0.2,0]]
Ki = [[1/800,0,-0.4], [0,1/800,-0.3], [0,0,1]]
F = [[0,0,0], [0,0,1], [0,-1,0]]
raw = mm(mm(transpose(Ki), E), Ki)
assert all(isclose(raw[i][j], .00025*F[i][j], abs_tol=1e-12)
           for i in range(3) for j in range(3))

def project(P, center_x):
    X,Y,Z = P
    assert Z > 0
    return [800*(X-center_x)/Z+320, 800*Y/Z+240, 1]

for P, expected_u2, depth in [([.5,.25,2],440,2), ([1,.5,4],480,4)]:
    p1,p2 = project(P,0),project(P,.2)
    assert p1 == [520,340,1]
    assert isclose(p2[0],expected_u2)
    assert abs(dot(p2,mv(F,p1))) < 1e-10
    assert abs(dot(mv(Ki,p2),mv(E,mv(Ki,p1)))) < 1e-10
    assert isclose(800*.2/(p1[0]-p2[0]), depth)

line = mv(F,[520,340,1])
def distance(p,l):
    return abs(dot(p,l))/hypot(l[0],l[1])
assert distance([400,340,1],line) == 0  # 올바른 대응임을 보장하지 않는다
assert distance([440,343,1],line) == 3
assert distance([440,343,1],[10*v for v in line]) == 3
~~~

Python 표준 라이브러리로 위 예제를 실행했다. F를 영상으로부터 추정하거나 RANSAC·특징점 검출을 실행한 것은 아니다.

</details>

## 7. 성립 조건과 실제 추정의 한계

기하가 알려져 있으면 E와 F를 계산할 수 있다. 알려져 있지 않으면 특징점 대응으로 F 또는 E를 추정한다. 이때 오대응을 견디는 추정 절차와 잔차 검사가 필요하다. OpenCV 예제는 특징 매칭 뒤 fundamental matrix를 추정하고 epiline을 시각화하는 흐름을 보여준다. [1] 사용한 알고리즘·좌표계·임계값 단위는 함께 기록해야 한다.

| 상황 | 주의할 점 |
| --- | --- |
| 순수 회전, $t=0$ | 위 정의에서 E=0이 되어 깊이를 제한할 baseline 정보가 없다 |
| 매우 작은 baseline·먼 점 | 시차가 작아 깊이가 작은 픽셀 오차에 민감하다 |
| 평면 위 점에 치우친 대응 | homography로도 설명될 수 있어 일반 3D 기하 추정에 주의한다 |
| 가림·반복 무늬 | 선 위에 후보가 없거나 여러 개일 수 있다 |
| 움직이는 물체 | 정적 장면의 한 상대 자세로 전체 대응을 설명하지 못할 수 있다 |

이상적 E는 rank 2와 두 개의 같은 비영 특잇값을 가진다. 하지만 E가 주어져도 이동의 절대 크기는 epipolar 제약만으로 결정되지 않는다. E에 배율을 곱해도 제약이 같기 때문이다. 앞에서 미터 단위 깊이를 계산할 수 있었던 것은 baseline 0.2 m를 외부 조건으로 주었기 때문이다.

## 8. 시사점과 확장 아이디어

**[해석]** 두 뷰 논문에서 “기하적으로 일관된 대응”은 “올바른 대응”보다 약한 주장이다. 동일한 선 위의 오대응이 남는지, 순수 회전·평면 장면에서 무엇을 가정하는지, 자세 정확도와 대응 정확도를 따로 확인하는지가 중요하다.

**[아이디어 · 미검증]** 같은 epipolar 거리 임계값이 보정 불확실성이 큰 영역에서도 적절할까?

- **가설:** 불확실성을 반영한 허용 폭이 고정 폭보다 올바른 대응을 덜 버릴 수 있다.
- **비교:** 고정 1·2·3 pixel 폭과 보정 파라미터를 반복 섭동해 얻은 위치별 허용 폭.
- **통제:** 같은 3D 점·카메라·오대응 집합을 사용하고, 왜곡 보정 좌표에서 비교한다.
- **최소 실험:** 합성 대응에 일정한 검출 노이즈를 넣고 별도로 K·자세의 추정 오차를 추가한다. 임계값 조정용 데이터와 평가 데이터를 분리한다.
- **지표:** 동일한 오대응 허용률에서 올바른 대응의 유지율을 비교하고 중심·가장자리별 결과를 기록한다.
- **기각 기준:** 이득이 단순히 허용 폭 확대나 오대응 증가로 설명되면 개선 주장을 기각한다. 실제 보정 오차를 대표하지 못하면 실측 일반화를 보류한다.

이 실험은 실행하지 않았다. 앞의 정확한 평행 스테레오 검산은 불확실성 모델의 성능 검증이 아니다.

## 9. 3분 복습

1. F에 첫 영상의 동차 점을 곱하면 무엇을 얻는가?
2. 왜 E와 F에 같은 좌표를 무조건 넣으면 안 되는가?
3. 예제에서 v=340을 만족하는 후보가 모두 정답이 아닌 이유는 무엇인가?

<details markdown="1">
<summary>답과 연결 설명</summary>

1. 두 번째 영상의 직선 계수 $l_2=Fp_1$을 얻는다.
2. E는 정규화 좌표, F는 해당 K와 일치하는 픽셀 좌표를 사용하기 때문이다.
3. 서로 다른 깊이 가설과 선 위의 오대응도 같은 제약을 만족할 수 있기 때문이다. 외관·추가 관측·가시성 검사가 필요하다.

</details>

## 출처와 다음 읽기

1. [OpenCV 4.13.0 — Epipolar Geometry](https://docs.opencv.org/4.13.0/da/de9/tutorial_py_epipolar_geometry.html), Basic Concepts와 매칭·epiline 계산 흐름.
2. [OpenCV 4.13.0 — calib3d](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html), computeCorrespondEpilines의 점-직선 방향과 계수 정규화.

확인일: 2026-11-21. 수치와 그림은 자체 작성했으며 Python으로 검증했다.

[이전: Camera Calibration]({{ '/notes/camera-calibration/' | relative_url }}) · [전체 학습 지도]({{ '/roadmap/' | relative_url }})

**다음 질문:** 대응 픽셀과 카메라가 주어지면 3D 점을 어떻게 구하고, 3D 점과 픽셀이 주어지면 카메라 자세를 어떻게 구하는가? Day 7은 Triangulation과 PnP다.
