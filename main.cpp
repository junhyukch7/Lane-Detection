#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream> 
#include <math.h> 
#include <algorithm>

using namespace std;
using namespace cv;

Mat gray_scale(Mat& image) {
	//--------------GRAYSCALE IMAGE-----------------
	// Define grayscale image
	Mat gray;

	// color to gray
	cvtColor(image, gray, COLOR_BGR2GRAY);

	imshow("gray scale", gray);
	waitKey(0); // wait for a key press
	return gray;

}
Mat Gaussian_blurring(Mat& gray) {
	int kSize = 9; // Guassian kenrnel size bigger kernel = more smoothing
	// Define smoothed image
	Mat blur;
	GaussianBlur(gray, blur, Size(kSize, kSize), 0, 0);

	imshow("bluring", blur);
	waitKey(0); // wait for a key press
	return blur;
}
Mat canny_edge(Mat& blur) {
	int minVal = 50;
	int maxVal = 150;

	// Define edge detection image, do edge detection
	Mat edge;
	Canny(blur, edge, minVal, maxVal);

	imshow("canny edge", edge);
	waitKey(0); // wait for a key press
	return edge;
}
Mat Roi(Mat& image, Mat& edge) {
	// roi를 만들기 위해 먼저 mask 행렬 만들기
	Mat mask = Mat::zeros(image.rows, image.cols, CV_8UC1); // CV_8UC3 to make it a 3 channel

	// mask를 만들 사다리꼴의 점 배열
	Point mask_points[1][4];
	
	mask_points[0][0] = Point(50, image.size().height);
	mask_points[0][1] = Point(image.size().width / 2 - 45, image.size().height / 2 + 60);
	mask_points[0][2] = Point(image.size().width / 2 + 45, image.size().height / 2 + 60);
	mask_points[0][3] = Point(image.size().width - 50, image.size().height);

	const Point* ppt[1] = { mask_points[0] }; // mask_point의 주소값을 전달해야 함
	int npt[] = { 4 }; // 사다리꼴 꼭짓점의 개수
	fillPoly(mask, ppt, npt, 1, Scalar(255,255,255), 8);

	imshow("Mask", mask);
	waitKey(0);

	
	Mat roi = edge.clone();
	bitwise_and(edge, mask, roi); // mask된 영역만 추출

	imshow("ROI", roi);
	waitKey(0);
	return roi;
}
vector<Vec4i> hough_transform(Mat& image, Mat& roi) {
	
	double rho = 2;
	double theta = CV_PI / 180;
	int threshold = 33; // good value to test : 33
	int minLineLength = 40; 
	int maxLineGap = 100; 

	vector<Vec4i> lines; 
	HoughLinesP(roi, lines, rho, theta, threshold, minLineLength, maxLineGap);

	Mat allLinesIm = Mat::zeros(image.rows, image.cols, CV_8UC3); // 라인 색때문에 차원이 3개가 필요

	for (Vec4i l : lines) {
		line(allLinesIm, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, 8);
	}
	imshow("hough lines", allLinesIm);
	waitKey(0);
	return lines;
}
double median(vector<double> vec) {

	// 벡터으 ㅣ길이
	int vecSize = vec.size();

	// 에러 처리
	if (vecSize == 0) {
		throw domain_error("median of empty vector");
	}

	// 연산횟수를 줄이기 위해 정렬
	sort(vec.begin(), vec.end());

	int middle;
	double median;

	// 벡터의 길이가 짝수인 경우 중앙값 존재 X, 가운데 두개의 값의 평균을 취함
	if (vecSize % 2 == 0) {
		middle = vecSize / 2;
		median = (vec[middle - 1] + vec[middle]) / 2;
	}

	// 홀수인 경우는 중앙값이 존재하므로 중앙의 위치한 값이 중앙값임
	else {
		middle = vecSize / 2;
		median = vec[middle];
	}
	return median;
}
void seperate_line(vector<Vec4i> lines, vector<vector<double>>& slopePositiveLines, vector<vector<double>>& slopeNegativeLines) {
	// 행을 추가하기 위한 cnt
	int negCounter = 0;
	int posCounter = 0;

	// 모든 라인들에 대해 기울기를 검사하여 양수 음수 나누기
	for (size_t i = 0; i != lines.size(); ++i) {

		// 현재 라인 좌표 정보
		double x1 = lines[i][0];
		double y1 = lines[i][1];
		double x2 = lines[i][2];
		double y2 = lines[i][3];

		// 점과 점사이의 거리 공식 적용하여 직선의 길이 구하기
		double lineLength = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));

		// 라인의 길이가 roi 사다리꼴의 빗변의 길이 정도는 되야함
		if (lineLength > 30) {

			// 값이 무한대로 발산하는 것을 방지
			if (x2 != x1) {

				// 기울기 공식
				double slope = (y2 - y1) / (x2 - x1);

				// 기울기가 양이라면?? == 왼쪽 차선
				if (slope > 0) {

					// x축과 라인이 이루는 각도 찾기 이 각도를 기준으로 라인의 적합성 평가
					double tanTheta = tan((abs(y2 - y1)) / (abs(x2 - x1))); // tan(theta)  = 높이 / 밑변
					double angle = atan(tanTheta) * 180 / CV_PI;

					// roi가 사다리꼴이기 때문에 Lane의 x축과의 각도는 90도 혹은 0도가 될 수 없다.
					// 수직 수평 라인 성분 제거
					if (abs(angle) < 85 && abs(angle) > 20) {

						// 행렬의 형태로 바꾸기 -> 행 성분(라인 넘버) 추가
						slopeNegativeLines.resize(negCounter + 1);

						// [x1, y1, x2, y2, slope] ->열 추가
						slopeNegativeLines[negCounter].resize(5);

						slopeNegativeLines[negCounter][0] = x1;
						slopeNegativeLines[negCounter][1] = y1;
						slopeNegativeLines[negCounter][2] = x2;
						slopeNegativeLines[negCounter][3] = y2;
						slopeNegativeLines[negCounter][4] = -slope; // 양의 기울기이기 때문에 - 

						// counter ++
						negCounter++;
					}

				}

				// 기울기가 음인경우 == 오른쪽 차선
				if (slope < 0) {

					// x축과 라인이 이루는 각도 찾기 이 각도를 기준으로 라인의 적합성 평가
					double tanTheta = tan((abs(y2 - y1)) / (abs(x2 - x1))); // tan(theta)  = 높이 / 밑변
					double angle = atan(tanTheta) * 180 / CV_PI;

					// roi가 사다리꼴이기 때문에 Lane의 x축과의 각도는 90도 혹은 0도가 될 수 없다.
					// 수직 수평 라인 성분 제거
					if (abs(angle) < 85 && abs(angle) > 20) {
						// 행렬의 형태로 바꾸기 -> 행 성분(라인 넘버) 추가
						slopePositiveLines.resize(posCounter + 1);

						// [x1, y1, x2, y2, slope] ->열 성분 조정
						slopePositiveLines[posCounter].resize(5);

						// 행렬 요소 추가
						slopePositiveLines[posCounter][0] = x1;
						slopePositiveLines[posCounter][1] = y1;
						slopePositiveLines[posCounter][2] = x2;
						slopePositiveLines[posCounter][3] = y2;
						slopePositiveLines[posCounter][4] = -slope;

						// counter++
						posCounter++;

					}
				}
			}
		}
	}
}
vector<double> get_slope_mean(vector<vector<double>>& slopePositiveLines, vector<vector<double>>& slopeNegativeLines) {
	vector<double> SlopeMean; // first = positve, second = negative
	
	// 양의 라인의 평균값을 구하기 위해 양의 기울기 값을 모두 저장
	vector<double> positiveSlopes;
	for (unsigned int i = 0; i != slopePositiveLines.size(); ++i) {
		positiveSlopes.push_back(slopePositiveLines[i][4]);
	}

	// 기울기들의 중간값 계산
	sort(positiveSlopes.begin(), positiveSlopes.end()); // 편하게 중앙값 계산을 위해 정렬
	double posSlopeMedian; // define positive slope median
	posSlopeMedian = median(positiveSlopes);


	// 좋은 기울기를 찾기 위해 현재 값과 중앙값의 차이를 확인 한 후 차이가 적으면 ok!!
	vector<double> posSlopesGood;
	double posSum = 0.0; // sum so we'll be able to get mean

	// Loop through positive slopes and add the good ones
	for (size_t i = 0; i != positiveSlopes.size(); ++i) {

		// 만약 현재값과 중앙값과의 차이가 일정 임계값보다 적다면 ok! 임계값은 테스트
		if (abs(positiveSlopes[i] - posSlopeMedian) < 0.9) { //posSlopeMedian*0.2
			posSlopesGood.push_back(positiveSlopes[i]); // Add slope to posSlopesGood
			posSum += positiveSlopes[i];
		}
	}

	// 최종 양의 평균값 계산
	double posSlopeMean = posSum / posSlopesGood.size();

	////////////////////////////////////////////////////////////////////////

	// 음의 라인의 평균값을 구하기 위해 음의 기울기 값을 모두 저장
	vector<double> negativeSlopes;
	for (size_t i = 0; i != slopeNegativeLines.size(); ++i) {
		negativeSlopes.push_back(slopeNegativeLines[i][4]);
	}

	// 음의 기울기 중앙값 계산
	sort(negativeSlopes.begin(), negativeSlopes.end()); // sort vec
	double negSlopeMedian; // define negative slope median
	negSlopeMedian = median(negativeSlopes);

	// 좋은 기울기를 찾기 위해 현재 값과 중앙값의 차이를 확인 한 후 차이가 적으면 ok!!
	vector<double> negSlopesGood;
	double negSum = 0.0; // sum so we'll be able to get mean

	// Loop through positive slopes and add the good ones
	for (size_t i = 0; i != negativeSlopes.size(); ++i) {

		// 만약 현재값과 중앙값과의 차이가 일정 임계값보다 적다면 ok!
		if (abs(negativeSlopes[i] - negSlopeMedian) < 0.9) { // < negSlopeMedian*0.2
			negSlopesGood.push_back(negativeSlopes[i]); // Add slope to negSlopesGood
			negSum += negativeSlopes[i]; // add to sum
		}
	}

	// 최종 음의 평균값 계산
	double negSlopeMean = negSum / negSlopesGood.size();

	SlopeMean.push_back(posSlopeMean);
	SlopeMean.push_back(negSlopeMean);

	return SlopeMean;

}
vector<double> get_intercept_coord(Mat& image, vector<vector<double>>& slopePositiveLines, vector<vector<double>>& slopeNegativeLines) {
	vector<double> InterceptPos;
	
	// Positive Lines
	vector<double> xInterceptPos; // 교점을 모두 구함

	// 교점좌표 찾기
	for (size_t i = 0; i != slopePositiveLines.size(); ++i) {
		double x1 = slopePositiveLines[i][0]; // x value
		double y1 = image.rows - slopePositiveLines[i][1]; // y축 반전
		double slope = slopePositiveLines[i][4];
		double yIntercept = y1 - slope * x1; // b = y-ax, y절편
		double xIntercept = -yIntercept / slope; // y=ax+b=0, x = -b/a , x절편
		if (isnan(xIntercept) == 0) { // 발산 방지
			xInterceptPos.push_back(xIntercept);
		}
	}

	// x좌표들의 중앙값 계산
	double xIntPosMed = median(xInterceptPos);

	// 좋은 값을 추출하기 위해 중앙값 정리를 이용한 x좌표와 중앙값 정리를 하지 않은 x좌표들을 비교
	// 차이가 적으면 좋은 값!!
	vector<double> xIntPosGood;
	double xIntSum = 0; 

	// 비교를 위해 루프 한번 더(여기선 중앙값 연산 X)
	for (size_t i = 0; i != slopePositiveLines.size(); ++i) {
		double x1 = slopePositiveLines[i][0]; // x value
		double y1 = image.rows - slopePositiveLines[i][1]; // y축 반전되어 있음
		double slope = slopePositiveLines[i][4];
		double yIntercept = y1 - slope * x1; // b = y-ax, y절편
		double xIntercept = -yIntercept / slope; // y=ax+b=0, x = -b/a , x절편

		// 발산하지도 않고 중앙값과 크게 차이가 나지 않는다면 통과
		if (isnan(xIntercept) == 0 && abs(xIntercept - xIntPosMed) < 0.35*xIntPosMed) {
			xIntPosGood.push_back(xIntercept); // add to 'good' vector
			xIntSum += xIntercept;
		}
	}

	// 양의 기울기의 x좌표 평균값 계산
	double xInterceptPosMean = xIntSum / xIntPosGood.size();

	/*---------------------------------------------------------------------*/

	// Negative Lines
	vector<double> xInterceptNeg; //교점을 모두 구함

	// 교점좌표 찾기
	for (size_t i = 0; i != slopeNegativeLines.size(); ++i) {
		double x1 = slopeNegativeLines[i][0]; // x value
		double y1 = image.rows - slopeNegativeLines[i][1]; // y축 반전
		double slope = slopeNegativeLines[i][4];
		double yIntercept = y1 - slope * x1; // b = y-ax, y절편
		double xIntercept = -yIntercept / slope; // y=ax+b=0, x = -b/a , x절편
		if (isnan(xIntercept) == 0) { // 발산 방지
			xInterceptNeg.push_back(xIntercept); 
		}
	}

	// x좌표들의 중앙값 계산
	double xIntNegMed = median(xInterceptNeg);

	// 좋은 값을 추출하기 위해 중앙값 정리를 이용한 x좌표와 중앙값 정리를 하지 않은 x좌표들을 비교
	// 차이가 적으면 좋은 값!!
	vector<double> xIntNegGood;
	double xIntSumNeg = 0; 

	// 비교를 위해 루프 한번 더(여기선 중앙값 연산 X)
	for (size_t i = 0; i != slopeNegativeLines.size(); ++i) {
		double x1 = slopeNegativeLines[i][0]; // x value
		double y1 = image.rows - slopeNegativeLines[i][1]; // y축반전
		double slope = slopeNegativeLines[i][4];
		double yIntercept = y1 - slope * x1; // b = y-ax, y절편
		double xIntercept = -yIntercept / slope; // y=ax+b=0, x = -b/a , x절편

		// 발산하지도 않고 중앙값과 크게 차이가 나지 않는다면 통과
		if (isnan(xIntercept) == 0 && abs(xIntercept - xIntNegMed) < .35*xIntNegMed) {
			xIntNegGood.push_back(xIntercept); 
			xIntSumNeg += xIntercept;
		}
	}

	// 음의 기울기의 x좌표 평균값 계산
	double xInterceptNegMean = xIntSumNeg / xIntNegGood.size();
	InterceptPos.push_back(xInterceptPosMean);
	InterceptPos.push_back(xInterceptNegMean);

	return InterceptPos;
}
Mat drawing_lane(Mat& image, double posSlopeMean, double xInterceptPosMean, double negSlopeMean, double xInterceptNegMean) {
	
	Mat laneLineImage = image.clone();
	Mat laneFill = image.clone();

	// Positive Slope Line
	double slope = posSlopeMean;
	double x1 = xInterceptPosMean;
	int y1 = 0;
	double y2 = image.size().height - (image.size().height - image.size().height*0.35); // ROI사다리꼴 높이
	double x2 = (y2 - y1) / slope + x1;

	// Add positive slope line to image
	x1 = int(x1 + 0.5);
	x2 = int(x2 + 0.5);
	y1 = int(y1 + 0.5);
	y2 = int(y2 + 0.5);
	line(laneLineImage, Point(x1, image.size().height - y1), Point(x2, image.size().height - y2),Scalar(0, 255, 0), 3, 8);


	// Negative Slope Line
	slope = negSlopeMean;
	double x1N = xInterceptNegMean;
	int y1N = 0;
	double x2N = (y2 - y1N) / slope + x1N;

	// Add negative slope line to image
	x1N = int(x1N + 0.5);
	x2N = int(x2N + 0.5);
	y1N = int(y1N + 0.5);
	line(laneLineImage, Point(x1N, image.size().height - y1N), Point(x2N, image.size().height - y2), Scalar(0, 255, 0), 3, 8);

	// Plot positive and negative lane lines
	imshow("Lane lines on image", laneLineImage);
	waitKey(0); // wait for a key press

	// Image Blend
	Point v1 = Point(x1, image.size().height - y1);
	Point v2 = Point(x2, image.size().height - y2);
	Point v3 = Point(x1N, image.size().height - y1N);
	Point v4 = Point(x2N, image.size().height - y2);

	// create vector from array of corner points of lane
	Point verticesBlend[] = { v1,v3,v4,v2 };
	vector<Point> verticesVecBlend(verticesBlend, verticesBlend + sizeof(verticesBlend) / sizeof(Point));

	// Create vector of vectors to be used in fillPoly, add the vertices we defined above
	vector<vector<Point> > verticesfp;
	verticesfp.push_back(verticesVecBlend);

	// Fill area created from vector points
	fillPoly(laneFill, verticesfp, Scalar(0, 255, 255));

	// Blend image
	float opacity = 0.25;
	Mat blendedIm;
	addWeighted(laneFill, opacity, image, 1 - opacity, 0, blendedIm);

	// Plot lane lines
	line(blendedIm, Point(x1, image.size().height - y1), Point(x2, image.size().height - y2), Scalar(0, 255, 0), 8, 8);
	line(blendedIm, Point(x1N, image.size().height - y1N), Point(x2N, image.size().height - y2), Scalar(0, 255, 0), 8, 8);

	// Show final frame
	imshow("Final Output", blendedIm);
	waitKey(0);
	return blendedIm;
}
void fillter_colors(Mat image, Mat& img_filtered) {
	Mat img_bgr = image.clone();
	Mat img_hsv, img_combine;
	Mat white_mask, white_image;
	Mat yellow_mask, yellow_image;

	//차선 색깔 범위 
	Scalar lower_white = Scalar(200, 200, 200); //흰색 차선 (RGB)
	Scalar upper_white = Scalar(255, 255, 255);
	Scalar lower_yellow = Scalar(10, 100, 100); //노란색 차선 (HSV)
	Scalar upper_yellow = Scalar(40, 255, 255);

	//Filter white pixels
	inRange(img_bgr, lower_white, upper_white, white_mask); // 흰색 범위값
	bitwise_and(img_bgr, img_bgr, white_image, white_mask); // input 이미지내에 해당 경계값 내에 있는 값만 추출

	cvtColor(img_bgr, img_hsv, COLOR_BGR2HSV);

	//Filter yellow pixels
	inRange(img_hsv, lower_yellow, upper_yellow, yellow_mask);
	bitwise_and(img_bgr, img_bgr, yellow_image, yellow_mask);


	//Combine the two above images
	addWeighted(white_image, 1.0, yellow_image, 1.0, 0.0, img_combine);

	img_filtered = img_combine.clone();
	imshow("filtered image", img_filtered);
	waitKey(0);
}
int main() {
	// image
	Mat image = imread("test1.jpg");
	
	if (image.empty()) 
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	imshow("Original Image", image);
	waitKey(0); 

	//Mat img_filtered;
	//fillter_colors(image, img_filtered); // 선 구분하기 힘든 경우 HSV로 변환하자
	
	Mat gray = gray_scale(image); // gray scale
	Mat blur = Gaussian_blurring(gray); // gaussian filtering
	Mat edge = canny_edge(blur); // canny edge detection
	Mat roi = Roi(image, edge); // roi
	vector<Vec4i> lines = hough_transform(image, roi); // hoguh transform
	
	
	// 이때 그어진 여러 라인 중에 최적화된 라인을 선택하기 위해 
	// 라인들의 중앙값이 최적의 해라고 가정하고 중앙 라인의 좌표를 구하자!!
	// 구해야 할 것 : x,y좌표 중앙값 기울기 중앙값

	// 좌측 라인과 우측 라인을 나누기 위해 기울기를 기준으로 나눈다.
	vector< vector<double> > slopePositiveLines; // [x1 y1 x2 y2 slope]
	vector< vector<double> > slopeNegativeLines; // [x1 y1 x2 y2 slope]
	seperate_line(lines, slopePositiveLines, slopeNegativeLines);

	// 나누어진 벡터들의 기울기 중앙값을 구한다.
	vector<double> SlopeMean = get_slope_mean(slopePositiveLines, slopeNegativeLines); 
	double posSlopeMean = SlopeMean[0]; // first = positive
	double negSlopeMean = SlopeMean[1]; // second = negative

	// 나누어진 벡터들의 좌표 중앙값 구하기
	vector<double> InterceptPos = get_intercept_coord(image, slopePositiveLines, slopeNegativeLines);
	double xInterceptPosMean = InterceptPos[0]; // first = positive
	double xInterceptNegMean = InterceptPos[1]; // second = negative

	// 평균 y교점 좌표가 없는 이유는 roi사다리꼴의 높이가 곧 y좌표이기 때문
	drawing_lane(image, posSlopeMean, xInterceptPosMean, negSlopeMean, xInterceptNegMean);
	
	/*
	// video
	VideoCapture cap("solidYellowLeft.mp4");
	if (!cap.isOpened()) {
		cerr << "fail to load video" << endl;
		return -1;
	}
	cout << "Frame count : " << cvRound(cap.get(CAP_PROP_FRAME_COUNT))<<endl;

	int w = cvRound(cap.get(CAP_PROP_FRAME_WIDTH));
	int h = cvRound(cap.get(CAP_PROP_FRAME_HEIGHT));
	double fps = cap.get(CAP_PROP_FPS);

	int fourcc = VideoWriter::fourcc('D', 'I', 'V', 'X');
	cout << "FPS : " << fps << endl;

	int delay = cvRound(1000 / fps);
	VideoWriter outputvideo("solidYellowLeft_result.avi", fourcc, fps, Size(w, h));
	
	if (!outputvideo.isOpened()) {
		cout << "file open failed" << endl;
		return -1;
	}
	Mat image;
	while (true) {
		cap >> image;
		if (image.empty()) {
			break;
		}
		Mat img_filtered;
		fillter_colors(image, img_filtered); // 선 구분하기 힘든 경우 HSV로 변환하자

		Mat gray = gray_scale(img_filtered); // gray scale
		Mat blur = Gaussian_blurring(gray); // gaussian filtering
		Mat edge = canny_edge(blur); // canny edge detection
		Mat roi = Roi(image, edge); // roi
		vector<Vec4i> lines = hough_transform(image, roi); // hoguh transform


		// 이때 그어진 여러 라인 중에 최적화된 라인을 선택하기 위해 
		// 라인들의 중앙값이 최적의 해라고 가정하고 중앙 라인의 좌표를 구하자!!
		// 구해야 할 것 : x,y좌표 중앙값 기울기 중앙값

		// 좌측 라인과 우측 라인을 나누기 위해 기울기를 기준으로 나눈다.
		vector< vector<double> > slopePositiveLines; // [x1 y1 x2 y2 slope]
		vector< vector<double> > slopeNegativeLines; // [x1 y1 x2 y2 slope]
		seperate_line(lines, slopePositiveLines, slopeNegativeLines);

		// 나누어진 벡터들의 기울기 중앙값을 구한다.
		vector<double> SlopeMean = get_slope_mean(slopePositiveLines, slopeNegativeLines);
		double posSlopeMean = SlopeMean[0]; // first = positive
		double negSlopeMean = SlopeMean[1]; // second = negative

		// 나누어진 벡터들의 좌표 중앙값 구하기
		vector<double> InterceptPos = get_intercept_coord(image, slopePositiveLines, slopeNegativeLines);
		double xInterceptPosMean = InterceptPos[0]; // first = positive
		double xInterceptNegMean = InterceptPos[1]; // second = negative

		// 동영상 저장시 drawing_lane return void에서 Mat으로
		Mat final;
		final = drawing_lane(image, posSlopeMean, xInterceptPosMean, negSlopeMean, xInterceptNegMean);
		outputvideo << final;
		if (waitKey(1) == 27) {
			break;
		}
	}
	destroyAllWindows();
	cap.release();
	*/
	destroyAllWindows();
	return 0;
	}