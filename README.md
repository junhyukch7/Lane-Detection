## OPENCV 기반 직선차선인식 알고리즘
<img src = "https://github.com/junhyukch7/Lane-Detection/blob/main/test%20image/Lane_detection%201.PNG" width="50%">

### Pipeline

#### Step 1 : 전처리 과정

1. filtering color (RGB to HSV, masking yellow and white)

<img src = "https://github.com/junhyukch7/Lane-Detection/blob/main/test%20image/test1_filtered_bad.PNG" width="50%">

* 차선의 색깔이 희미해지거나 주변색과 유사해질 경우 위와 같이 차선을 인식하지 못하게 된다.

<img src = "https://github.com/junhyukch7/Lane-Detection/blob/main/test%20image/test1_filtered.PNG" width="50%"><img src = "https://github.com/junhyukch7/Lane-Detection/blob/main/test%20image/test1_filtered_good.PNG" width="50%">

* 따라서 색을 표현하는 방법을 기존의 RGB에서 HSV로 바꾼다. HSV 이미지에서는 H(Hue)가 일정한 범위를 갖는 순수한 색 정보를 가지고 있기 때문에 RGB 이미지보다 쉽게 색을 분류할 수 있다.
또한 선을 분류할 때 연산량을 줄이기 위해 차선색깔 필터(흰색 노란색)를 적용하였다.

2. Gray Scale

* 해당 과정은 filtering color단계에서 충분히 전처리 과정이 수행되어 적용하지 않아도 큰 차이를 보이지 않았다. 따라서 filtering color를 적용하는 경우 해당 과정은 연산량 감소를 위해 건너뛰어도 무방하다.

3. Gaussinan filtering
4. Canny edge transform
5. Set Roi
6. hough transform

#### Step 2 : 대표선 찾기

아이디어 : 중앙값과 비교하여 중앙값의 가까운 값들의 평균이 대표선

1. seperate_line()
* 선의 기울기를 기준으로 왼쪽차선과 오른쪽 차선을 구분한다.
* 이때 수평의 성분과 수직의 성분은 제외한다.

2. get slope mean()
* 각각의 차선의 기울기의 중앙값을 구한 후 hough transfrom으로 얻어진 직선과 비교하여 오차가 작은 기울기의 집합을 모은다.
* 모아진 직선의 평균이 기울기 값이 이상적인 기울기 값

3. get intercept coord()
* 기울기 구할 때와 마찬가지로 x좌표의 중앙값을 구한 후 모든 x좌표와 비교하여 오차가 작은 좌표의 집합을 모은다.
* 모아진 좌표의 평균이 이상적인 평균 좌표 위치

#### Step 3 : 선 그리기
* cv::line(), cv::fillPoly()함수를 이용하여 이상적인 차선과 영역을 그린다.
* 이때 평균y좌표는 ROI사다리꼴의 높이 좌표와 같다
---
### Result video
<img width="50%" src="https://github.com/junhyukch7/Lane-Detection/blob/main/test%20image/solidyellow.gif">

<img width="50%" src="https://github.com/junhyukch7/Lane-Detection/blob/main/test%20image/challenge.gif">

---
### Conclusion
* 결과 영상에서도 볼 수 있듯이 직선에서의 차선인식은 그림자나 차선의 경계가 희미해지는 영역에서도 잘 인식하는 것을 볼 수 있다. 하지만 곡선의 영역에서 인식이 정확하게 되지 않음을 확인할 수 있었다. 이를 위해 perspective 변환을 이용하여 곡선의 차선을 검출한 후 비선형회귀분석 알고리즘을 적용하면 더 나은 결과를 얻을 수 있을 것으로 생각한다. 또한 Pipeline에서 ROI를 추출하는 과정을 첫번째로 한다면 연산량을 더 줄일 수 있을 것이다. 기회가 된다면 해당 문제점들을 보완하여 다음 프로젝트에서는 곡선차선인식 알고리즘에 대해 다뤄볼 예정이다.
