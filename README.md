# Lane-Detection
## opencv 기반 직선차선인식 알고리즘

![solidyellow](https://user-images.githubusercontent.com/79674592/117899866-02e88a00-b303-11eb-9714-3bbcc257f100.gif)

### Pipeline

#### Step 1 : 전처리 과정

1. filtering color

차선의 색깔이 희미해지거나 주변색과 유사해질 경우 다음과 같이 차선을 인식하지 못하게 된다.

<img src = "https://github.com/junhyukch7/Lane-Detection/blob/main/test%20image/test1_filtered_bad.PNG" width="40%">

따라서 색을 표현하는 방법을 기존의 RGB에서 HSV로 바꾼다. HSV 이미지에서는 H(Hue)가 일정한 범위를 갖는 순수한 색 정보를 가지고 있기 때문에 RGB 이미지보다 쉽게 색을 분류할 수 있다.
또한 선을 분류할 때 연산량을 줄이기 위해 차선색깔 필터(흰색 노란색)를 적용하였다.

<img src = "https://github.com/junhyukch7/Lane-Detection/blob/main/test%20image/test1_filtered.PNG" width="40%"><img src = "https://github.com/junhyukch7/Lane-Detection/blob/main/test%20image/test1_filtered_good.PNG" width="40%">

2. Gray Scale

해당 과정은 filtering color단계에서 충분히 전처리 과정이 수행되어 적용하지 않아도 큰 차이를 보이지 않았다. 따라서 filtering color를 적용하는 경우 해당 과정은 연산량 감소를 위해 건너뛰어도 무방하다.

3. Gaussinan filtering
4. Canny edge transform
5. Set Roi
6. hough transform

#### Step 2 : 대표선 찾기
