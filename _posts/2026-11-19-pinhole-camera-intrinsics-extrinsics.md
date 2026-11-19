---
layout: post
title: "Pinhole camera와 K, R, t: 3D 점은 어떻게 픽셀이 되는가"
description: "세계 좌표를 카메라 좌표로 옮기고 깊이로 나눈 뒤 픽셀로 바꾼다. 투영과 역투영을 같은 예제로 확인한다."
date: 2026-11-19 05:00:00 +09:00
category_label: "기초 개념"
series_day: 4
topic: "CAMERA GEOMETRY"
cover: /assets/day04-camera-ray.svg
tags: [pinhole, intrinsics, extrinsics, projection, 카메라]
reading_time: 10
---

> **1분 요약**
>
> 이상적인 pinhole 모델에서 투영은 **세계 → 카메라 좌표 변환**, **깊이로 나누기**, **픽셀 좌표로 바꾸기**의 세 단계다. $R,t$는 첫 단계를, $K$는 마지막 단계를 담당한다. World-to-camera 규약에서 카메라 중심은 $t$가 아니라 $C_w=-R^\top t$다. 한 픽셀을 거꾸로 따라가면 하나의 3D 점이 아니라 광선을 얻는다.
>
> **기억할 문장:** 행렬 이름보다 변환 방향을 먼저 확인하고, 나누기 전의 깊이를 보존한다.

## 1. 전체 내용: 픽셀 이전의 좌표계

[Day 3]({{ '/notes/convolution-feature-maps/' | relative_url }})에서는 픽셀 배열에 필터를 적용했다. 오늘은 그 픽셀이 3차원 장면과 어떻게 연결되는지 다룬다. 공간의 한 점은 세계 좌표와 카메라 좌표로 각각 표현할 수 있다. 카메라 좌표의 가로·세로 성분을 깊이로 나누면 광선 방향을 나타내는 정규화 좌표가 된다. 여기에 초점거리와 주점을 적용하면 이미지의 연속적인 픽셀 좌표를 얻는다. 이 과정에서 깊이를 잃으므로 한 장의 좌표만으로 원래 3D 위치를 정할 수 없다.

이 글은 **왜곡 없음, skew 0, 단일 투영 중심**을 가정한다. 흐림·가림·rolling shutter·렌즈 왜곡은 모델링하지 않는다. 수치·도식·검증 코드는 직접 작성했다. 기본 투영식은 OpenCV 공식 문서 [1], 자세 방향과 카메라 축 규약은 COLMAP 공식 문서 [2]를 확인했다.

| 기호 | 의미 | 크기·단위 |
| --- | --- | --- |
| $\mathbf P_w$ | 세계 좌표의 점 | $3\times1$, m |
| $\mathbf P_c=(X_c,Y_c,Z_c)^\top$ | 카메라 좌표의 같은 점 | $3\times1$, m |
| $R,t$ | 세계 → 카메라 회전·이동 | $3\times3$, $3\times1$; $t$는 m |
| $K$ | 내부 파라미터 행렬 | $3\times3$ |
| $(u,v)$ | 픽셀 좌표 | pixel |
| $(x_n,y_n)$ | 정규화 이미지 좌표 | 무차원 |

벡터와 그 성분을 구별하기 위해 점 벡터에는 굵은 $\mathbf P$를 사용한다. 이미지에서 $u$는 오른쪽, $v$는 아래로 증가한다. 카메라의 $X_c,Y_c,Z_c$ 축은 각각 오른쪽·아래·앞쪽으로 둔다. 이는 COLMAP 문서의 규약과 같다. [2] 다른 라이브러리를 연결할 때는 축과 handedness를 별도로 확인한다.

## 2. 첫 단계: 점을 카메라의 기준으로 표현한다

World-to-camera 변환은 다음과 같다.

$$
\mathbf P_c=R\mathbf P_w+t
$$

$R$은 회전행렬이므로 $R^\top R=I$, $\det R=1$이다. 세계 좌표로 표현한 카메라 중심을 $C_w$라 하면 카메라 좌표에서는 원점이다. 따라서 다음을 얻는다.

$$
0=RC_w+t,\qquad C_w=-R^\top t
$$

이를 원래 식에 넣으면 $\mathbf P_c=R(\mathbf P_w-C_w)$다. 먼저 카메라 중심을 빼고, 그 상대 위치를 카메라 축으로 표현한다고 읽을 수 있다. $t$는 **세계 원점을 카메라 좌표로 표현한 값**이지, 세계 좌표에서의 카메라 위치가 아니다.

역변환은 $\mathbf P_w=R^\top\mathbf P_c+C_w$다. 동차변환으로 쓰면 다음과 같다.

$$
T_{c\leftarrow w}=
\begin{bmatrix}R&t\\0^\top&1\end{bmatrix},
\qquad
T_{w\leftarrow c}=
\begin{bmatrix}R^\top&-R^\top t\\0^\top&1\end{bmatrix}
$$

**[해석]** 데이터에 “pose”라고만 적혀 있으면 바로 곱하지 않는다. World-to-camera인지 camera-to-world인지, 행벡터인지 열벡터인지부터 찾는다. 예를 들어 COLMAP의 images.txt 자세는 world-to-camera다. [2]

## 3. 두 번째·세 번째 단계: 깊이와 픽셀 단위를 구별한다

카메라 앞의 점에 대해 $Z_c>0$을 요구한다. 정규화 좌표는 다음과 같다.

$$
x_n=\frac{X_c}{Z_c},\qquad y_n=\frac{Y_c}{Z_c}
$$

그다음 픽셀 단위 초점거리 $f_x,f_y$와 주점 $c_x,c_y$를 적용한다.

$$
K=\begin{bmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix},
\qquad
\begin{bmatrix}u\\v\\1\end{bmatrix}
=K\begin{bmatrix}x_n\\y_n\\1\end{bmatrix}
$$

즉 $u=f_xX_c/Z_c+c_x$, $v=f_yY_c/Z_c+c_y$다. 세 단계를 합친 동차식은 아래와 같다. [1]

$$
Z_c\begin{bmatrix}u\\v\\1\end{bmatrix}
=K\begin{bmatrix}R&t\end{bmatrix}
\begin{bmatrix}\mathbf P_w\\1\end{bmatrix}
$$

차원은 $(3\times3)(3\times4)(4\times1)=(3\times1)$이다. 오른쪽 결과를 그대로 $(u,v,1)$로 읽으면 안 된다. 마지막 성분으로 나누어야 한다. $Z_c=0$에서는 이 투영을 정의할 수 없고, $Z_c<0$인 점은 이 글의 전방 카메라 관측 대상에서 제외한다.

$f_x=800$이라는 값은 800 mm가 아니다. 픽셀 좌표와 연결된 초점거리다. 또한 $K$는 원근 나눗셈 자체를 대신하지 않는다. 이를 구별하면 “행렬곱은 맞는데 픽셀이 틀린” 오류를 찾기 쉽다.

## 4. 회전과 이동이 있는 숫자 예제

가상의 카메라를 다음처럼 설정한다. 영상 크기는 $640\times480$, 왜곡은 없으며 모든 공간 길이는 m다.

$$
K=\begin{bmatrix}800&0&320\\0&600&240\\0&0&1\end{bmatrix},
\quad
R=\begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix},
\quad C_w=\begin{bmatrix}1\\0\\0\end{bmatrix}
$$

이때 $t=-RC_w=(0,-1,0)^\top$다. 점 $\mathbf P_w=(2,1,4)^\top$를 투영하면 다음과 같다.

| 단계 | 계산 결과 |
| --- | --- |
| 카메라 좌표 | $R\mathbf P_w+t=(-1,1,4)^\top$ |
| 정규화 좌표 | $(-1/4,1/4)=(-0.25,0.25)$ |
| 픽셀 좌표 | $(800(-0.25)+320,\ 600(0.25)+240)=(120,390)$ |

검산으로 $R^\top(-1,1,4)^\top+C_w=(2,1,4)^\top$를 얻는다. 카메라 중심도 $RC_w+t=0$을 만족한다.

이 예제에서 $t$ 자리에 잘못 $C_w$를 넣으면 카메라 좌표는 $(0,2,4)^\top$, 픽셀은 $(320,540)$이 된다. 올바른 값과의 차이는 $(200,150)$ pixel, 2D 거리는 250 pixel이다. 이는 규약 오류를 보여주는 계산이며 특정 추정기의 성능 수치가 아니다.

## 5. 그림으로 보는 깊이의 손실과 역투영

<figure>
<img src="{{ '/assets/day04-camera-ray.svg' | relative_url }}" alt="카메라 중심에서 같은 광선 위에 있는 카메라 좌표 (X,Z)=(-1,4), (-2,8)은 정규화 평면 Z=1의 동일한 위치 X=-0.25로 투영된다.">
<figcaption>그림 1. 직접 작성한 카메라 좌표의 X–Z 단면. 가로는 Z, 아래 방향은 양의 X다. 가상의 정규화 평면 Z=1을 사용하며 실제 센서 위치를 그린 것이 아니다. Y 성분은 생략했다.</figcaption>
</figure>

그림의 두 점은 같은 직선 위에 있고 $X_c/Z_c=-0.25$가 같다. 본문 예제의 전체 3D 점 $(-1,1,4)$와 $(-2,2,8)$도 같은 픽셀 $(120,390)$로 간다.

반대로 왜곡 없는 픽셀 $\tilde p=(u,v,1)^\top$에서 다음 벡터를 얻는다.

$$
d_c=K^{-1}\tilde p,\qquad
\mathbf P_w(\lambda)=C_w+\lambda R^\top d_c,\quad \lambda>0
$$

$d_c$의 세 번째 성분을 1로 둔 현재 규약에서 $\lambda$는 **카메라 Z 깊이**다. 예제에서는 $d_c=(-0.25,0.25,1)^\top$, $R^\top d_c=(0.25,0.25,1)^\top$다.

| 선택한 깊이 $\lambda$ | 세계 좌표 | 투영 픽셀 |
| --- | --- | --- |
| 4 m | $(2,1,4)$ | $(120,390)$ |
| 8 m | $(3,2,8)$ | $(120,390)$ |

**[해석]** $K^{-1}$은 픽셀을 3D 점으로 복원하는 마법의 역행렬이 아니다. 방향을 되찾을 뿐이며, 광선의 어느 지점인지는 추가 깊이·다중 시점·장면 제약이 필요하다.

<details markdown="1">
<summary>Z 깊이와 카메라까지의 거리는 다르다</summary>

$\mathbf P_c=(-1,1,4)$의 Z 깊이는 4 m지만 카메라 중심까지의 유클리드 거리는 $\sqrt{1+1+16}=\sqrt{18}\approx4.2426$ m다. $d_c$를 단위벡터로 정규화하면 광선 계수는 거리로 바뀐다. 깊이 데이터셋을 읽을 때 Z-depth인지 range인지 확인해야 한다.

또한 $\mathbf P_c$ 전체에 같은 양의 배수를 곱하면 투영 픽셀은 유지된다. 이는 **카메라 중심을 기준으로 광선을 따라 이동하는 경우**다. 세계 원점 기준으로 점만 배율 조정하고 카메라 위치를 그대로 두는 경우와 혼동하지 않는다.

</details>

## 6. Crop과 resize 뒤의 K: 좌표 변환으로 유도한다

이미지 처리 뒤의 픽셀 좌표가 $\tilde p'=A\tilde p$를 따른다면 새 내부 행렬은 $K'=AK$다. 이는 $\tilde p=Kd_c$에 대입해 바로 얻는 관계다.

원본 좌표에서 왼쪽 80 pixel, 위쪽 40 pixel을 잘라내는 crop이면 $u'=u-80,\ v'=v-40$이다. 따라서 주점은 $(240,200)$으로 바뀌고 예제의 점은 $(40,350)$이 된다. 초점거리는 유지된다.

이 crop 결과에 **좌표 매핑을 $u''=0.5u',v''=0.5v'$로 정의한** 절반 축소를 적용하면 다음과 같다.

$$
A=\begin{bmatrix}0.5&0&-40\\0&0.5&-20\\0&0&1\end{bmatrix},
\quad
K'=\begin{bmatrix}400&0&120\\0&300&100\\0&0&1\end{bmatrix}
$$

새 투영점은 $(20,175)$다. 이것은 명시한 좌표 매핑에 대한 계산이다. 실제 resize API가 픽셀 중심을 $u''=s(u'+0.5)-0.5$로 매핑한다면 추가 상수항이 생긴다. 사용하는 샘플링 규칙에서 $A$를 정해야 하며, 단순 배율만 무조건 적용하지 않는다.

## 7. 실행한 수치 검증과 모델의 한계

<details markdown="1">
<summary>Python으로 투영·역변환·광선·crop/resize 확인</summary>

~~~python
from math import isclose, sqrt

def mv(a, x):
    return [sum(v * z for v, z in zip(row, x)) for row in a]

def add(a, b):
    return [x + y for x, y in zip(a, b)]

def near(a, b):
    return all(isclose(x, y, abs_tol=1e-9) for x, y in zip(a, b))

R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
RT = [list(row) for row in zip(*R)]
C = [1, 0, 0]
t = [-v for v in mv(R, C)]
K = [[800, 0, 320], [0, 600, 240], [0, 0, 1]]

def project(p, intrinsic=K, translation=t):
    pc = add(mv(R, p), translation)
    assert pc[2] > 0
    q = mv(intrinsic, pc)
    return [q[0] / q[2], q[1] / q[2]]

p = [2, 1, 4]
pc = add(mv(R, p), t)
assert near(pc, [-1, 1, 4])
assert near(project(p), [120, 390])
assert near(add(mv(RT, pc), C), p)
assert near(add(mv(R, C), t), [0, 0, 0])
assert near(project(p, translation=C), [320, 540])
assert isclose(sqrt(200**2 + 150**2), 250)

dc = [(120 - 320) / 800, (390 - 240) / 600, 1]
dw = mv(RT, dc)
for depth in (4, 8):
    point = add(C, [depth * v for v in dw])
    assert near(project(point), [120, 390])

K2 = [[400, 0, 120], [0, 300, 100], [0, 0, 1]]
assert near(project(p, intrinsic=K2), [20, 175])
assert isclose(sqrt(sum(v*v for v in pc)), sqrt(18))
~~~

Python 표준 라이브러리로 위 검사를 실행했다. OpenCV·COLMAP의 추정 알고리즘을 실행한 것은 아니다. 실측 영상, 노이즈, 렌즈 왜곡이 없는 합성 계산에 한정한다.

</details>

| 혼동 | 확인할 기준 |
| --- | --- |
| $t$가 카메라의 세계 위치다 | 변환 방향과 $C_w=-R^\top t$를 확인한다 |
| $K$만 곱하면 2D 좌표가 완성된다 | 동차 좌표의 마지막 성분으로 나눈다 |
| 픽셀 하나에서 깊이를 복원한다 | 역투영은 광선이며 깊이가 추가로 필요하다 |
| 보이는 점이면 항상 이미지 안에 있다 | $Z_c>0$ 외에도 영상 범위와 가림을 확인한다 |
| 모든 이미지에서 같은 K를 쓴다 | crop·resize·초점 변화와 카메라 모델을 확인한다 |

투영식은 점이 표면 뒤에 가려지는지 알려주지 않는다. 양의 깊이와 영상 범위만으로 실제 가시성을 보장할 수 없다. 왜곡된 원본 픽셀을 그대로 $K^{-1}$에 넣는 것도 이상적인 광선 역투영과 다르다. 왜곡을 모델링하거나 먼저 보정한 좌표를 사용해야 한다. [1]

## 8. 시사점과 확장 아이디어

**[해석]** 3D 재구성·NeRF·다중 시점 논문을 읽을 때는 “카메라가 주어진다”를 행렬 파일의 존재로만 읽지 않는다. 자세 방향, 이미지 전처리와 K의 일치, 깊이 단위를 함께 확인한다. 모델 개선처럼 보이는 차이가 전처리 좌표 불일치에서 발생할 수 있으므로 입력 조건부터 맞추는 것이 유용하다.

**[아이디어 · 미검증]** 이미지 crop·resize의 K 갱신 누락을 재투영 잔차의 모양으로 구별할 수 있을까?

- **가설:** crop에 따른 주점 이동을 누락하면 거의 일정한 잔차가, 초점거리 배율을 누락하면 주점에서의 위치에 비례하는 잔차가 나타난다.
- **비교:** 올바른 $K'=AK$, 기존 K 유지, 주점만 갱신한 조건.
- **통제:** 동일한 3D 점·자세·좌표 매핑·가시점 집합을 사용한다. 처음에는 왜곡과 검출 노이즈를 제외한다.
- **최소 실험:** 여러 깊이와 화면 위치의 점을 투영하고 변환한 정답 좌표와 각 조건의 예측을 비교한다.
- **지표:** 픽셀 RMS와 잔차 벡터장의 공간 패턴, 오차와 주점 거리의 관계를 기록한다.
- **기각 기준:** 단순 합성 조건에서도 예상 패턴이 분리되지 않으면 변환 순서와 좌표 규약을 다시 점검한다. 노이즈·왜곡을 넣었을 때 구별되지 않으면 실제 오류 진단으로 일반화하지 않는다.

위 확장 실험은 실행하지 않았다. 앞의 한 점 계산은 좌표 변환 검산이며 오류 분류기의 검증이 아니다.

## 9. 3분 복습

1. World-to-camera 식에서 카메라 중심은 왜 $-R^\top t$인가?
2. 픽셀 $(120,390)$을 $K^{-1}$로 역투영하면 무엇이 정해지고 무엇이 남는가?
3. 예제에서 crop 후 좌표를 절반으로 줄였을 때 주점과 픽셀 위치는 무엇인가?

<details markdown="1">
<summary>답과 연결 설명</summary>

1. 카메라 중심은 카메라 좌표에서 0이므로 $RC_w+t=0$을 푼 결과다.
2. 카메라 광선 방향 $(-0.25,0.25,1)$이 정해진다. 그 광선 위의 깊이는 남는다.
3. 이 글의 좌표 매핑에서 주점은 $(120,100)$, 투영점은 $(20,175)$다. 실제 API에서는 픽셀 중심 규약도 확인한다.

</details>

## 출처와 다음 읽기

1. [OpenCV 4.13.0 — Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html), Detailed Description의 왜곡 없는 투영식과 왜곡 모델.
2. [COLMAP — Output Format](https://colmap.github.io/format.html#images-txt), images.txt의 자세 방향·카메라 중심·축 규약.

확인일: 2026-11-19. 직접 작성한 도식과 합성 수치 예제를 사용했다.

[이전: Convolution과 feature map]({{ '/notes/convolution-feature-maps/' | relative_url }}) · [선수 지식: 벡터·행렬·동차좌표]({{ '/notes/vectors-matrices-homogeneous/' | relative_url }}) · [전체 학습 지도]({{ '/roadmap/' | relative_url }})

**다음 질문:** 투영 규칙을 알았다면 실제 카메라의 K와 왜곡 계수는 어떻게 구하는가? [Day 5 — Camera Calibration]({{ '/notes/camera-calibration/' | relative_url }})에서 이어간다.
