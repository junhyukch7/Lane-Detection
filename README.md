# Lane-Detection
## opencv 기반 직선차선인식 알고리즘

![solidyellow](https://user-images.githubusercontent.com/79674592/117899866-02e88a00-b303-11eb-9714-3bbcc257f100.gif)

### Pipeline

#### Step 1

영상 처리 단계 : 연산량을 줄이기 위해 이미지를 다음과 같은 절차를 거친다.

1. filtering color
차선의 색깔이 희미해지거나 주변색과 유사해질 경우 다음과 같이 차선을 인식하지 못하게 된다.

<img src = "https://github.com/junhyukch7/Lane-Detection/blob/main/test%20image/test1_filtered_bad.PNG" width="40%">



