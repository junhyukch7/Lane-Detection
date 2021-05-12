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
	// roi�� ����� ���� ���� mask ��� �����
	Mat mask = Mat::zeros(image.rows, image.cols, CV_8UC1); // CV_8UC3 to make it a 3 channel

	// mask�� ���� ��ٸ����� �� �迭
	Point mask_points[1][4];
	
	mask_points[0][0] = Point(50, image.size().height);
	mask_points[0][1] = Point(image.size().width / 2 - 45, image.size().height / 2 + 60);
	mask_points[0][2] = Point(image.size().width / 2 + 45, image.size().height / 2 + 60);
	mask_points[0][3] = Point(image.size().width - 50, image.size().height);

	const Point* ppt[1] = { mask_points[0] }; // mask_point�� �ּҰ��� �����ؾ� ��
	int npt[] = { 4 }; // ��ٸ��� �������� ����
	fillPoly(mask, ppt, npt, 1, Scalar(255,255,255), 8);

	imshow("Mask", mask);
	waitKey(0);

	
	Mat roi = edge.clone();
	bitwise_and(edge, mask, roi); // mask�� ������ ����

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

	Mat allLinesIm = Mat::zeros(image.rows, image.cols, CV_8UC3); // ���� �������� ������ 3���� �ʿ�

	for (Vec4i l : lines) {
		line(allLinesIm, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, 8);
	}
	imshow("hough lines", allLinesIm);
	waitKey(0);
	return lines;
}
double median(vector<double> vec) {

	// ������ �ӱ���
	int vecSize = vec.size();

	// ���� ó��
	if (vecSize == 0) {
		throw domain_error("median of empty vector");
	}

	// ����Ƚ���� ���̱� ���� ����
	sort(vec.begin(), vec.end());

	int middle;
	double median;

	// ������ ���̰� ¦���� ��� �߾Ӱ� ���� X, ��� �ΰ��� ���� ����� ����
	if (vecSize % 2 == 0) {
		middle = vecSize / 2;
		median = (vec[middle - 1] + vec[middle]) / 2;
	}

	// Ȧ���� ���� �߾Ӱ��� �����ϹǷ� �߾��� ��ġ�� ���� �߾Ӱ���
	else {
		middle = vecSize / 2;
		median = vec[middle];
	}
	return median;
}
void seperate_line(vector<Vec4i> lines, vector<vector<double>>& slopePositiveLines, vector<vector<double>>& slopeNegativeLines) {
	// ���� �߰��ϱ� ���� cnt
	int negCounter = 0;
	int posCounter = 0;

	// ��� ���ε鿡 ���� ���⸦ �˻��Ͽ� ��� ���� ������
	for (size_t i = 0; i != lines.size(); ++i) {

		// ���� ���� ��ǥ ����
		double x1 = lines[i][0];
		double y1 = lines[i][1];
		double x2 = lines[i][2];
		double y2 = lines[i][3];

		// ���� �������� �Ÿ� ���� �����Ͽ� ������ ���� ���ϱ�
		double lineLength = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));

		// ������ ���̰� roi ��ٸ����� ������ ���� ������ �Ǿ���
		if (lineLength > 30) {

			// ���� ���Ѵ�� �߻��ϴ� ���� ����
			if (x2 != x1) {

				// ���� ����
				double slope = (y2 - y1) / (x2 - x1);

				// ���Ⱑ ���̶��?? == ���� ����
				if (slope > 0) {

					// x��� ������ �̷�� ���� ã�� �� ������ �������� ������ ���ռ� ��
					double tanTheta = tan((abs(y2 - y1)) / (abs(x2 - x1))); // tan(theta)  = ���� / �غ�
					double angle = atan(tanTheta) * 180 / CV_PI;

					// roi�� ��ٸ����̱� ������ Lane�� x����� ������ 90�� Ȥ�� 0���� �� �� ����.
					// ���� ���� ���� ���� ����
					if (abs(angle) < 85 && abs(angle) > 20) {

						// ����� ���·� �ٲٱ� -> �� ����(���� �ѹ�) �߰�
						slopeNegativeLines.resize(negCounter + 1);

						// [x1, y1, x2, y2, slope] ->�� �߰�
						slopeNegativeLines[negCounter].resize(5);

						slopeNegativeLines[negCounter][0] = x1;
						slopeNegativeLines[negCounter][1] = y1;
						slopeNegativeLines[negCounter][2] = x2;
						slopeNegativeLines[negCounter][3] = y2;
						slopeNegativeLines[negCounter][4] = -slope; // ���� �����̱� ������ - 

						// counter ++
						negCounter++;
					}

				}

				// ���Ⱑ ���ΰ�� == ������ ����
				if (slope < 0) {

					// x��� ������ �̷�� ���� ã�� �� ������ �������� ������ ���ռ� ��
					double tanTheta = tan((abs(y2 - y1)) / (abs(x2 - x1))); // tan(theta)  = ���� / �غ�
					double angle = atan(tanTheta) * 180 / CV_PI;

					// roi�� ��ٸ����̱� ������ Lane�� x����� ������ 90�� Ȥ�� 0���� �� �� ����.
					// ���� ���� ���� ���� ����
					if (abs(angle) < 85 && abs(angle) > 20) {
						// ����� ���·� �ٲٱ� -> �� ����(���� �ѹ�) �߰�
						slopePositiveLines.resize(posCounter + 1);

						// [x1, y1, x2, y2, slope] ->�� ���� ����
						slopePositiveLines[posCounter].resize(5);

						// ��� ��� �߰�
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
	
	// ���� ������ ��հ��� ���ϱ� ���� ���� ���� ���� ��� ����
	vector<double> positiveSlopes;
	for (unsigned int i = 0; i != slopePositiveLines.size(); ++i) {
		positiveSlopes.push_back(slopePositiveLines[i][4]);
	}

	// ������� �߰��� ���
	sort(positiveSlopes.begin(), positiveSlopes.end()); // ���ϰ� �߾Ӱ� ����� ���� ����
	double posSlopeMedian; // define positive slope median
	posSlopeMedian = median(positiveSlopes);


	// ���� ���⸦ ã�� ���� ���� ���� �߾Ӱ��� ���̸� Ȯ�� �� �� ���̰� ������ ok!!
	vector<double> posSlopesGood;
	double posSum = 0.0; // sum so we'll be able to get mean

	// Loop through positive slopes and add the good ones
	for (size_t i = 0; i != positiveSlopes.size(); ++i) {

		// ���� ���簪�� �߾Ӱ����� ���̰� ���� �Ӱ谪���� ���ٸ� ok! �Ӱ谪�� �׽�Ʈ
		if (abs(positiveSlopes[i] - posSlopeMedian) < 0.9) { //posSlopeMedian*0.2
			posSlopesGood.push_back(positiveSlopes[i]); // Add slope to posSlopesGood
			posSum += positiveSlopes[i];
		}
	}

	// ���� ���� ��հ� ���
	double posSlopeMean = posSum / posSlopesGood.size();

	////////////////////////////////////////////////////////////////////////

	// ���� ������ ��հ��� ���ϱ� ���� ���� ���� ���� ��� ����
	vector<double> negativeSlopes;
	for (size_t i = 0; i != slopeNegativeLines.size(); ++i) {
		negativeSlopes.push_back(slopeNegativeLines[i][4]);
	}

	// ���� ���� �߾Ӱ� ���
	sort(negativeSlopes.begin(), negativeSlopes.end()); // sort vec
	double negSlopeMedian; // define negative slope median
	negSlopeMedian = median(negativeSlopes);

	// ���� ���⸦ ã�� ���� ���� ���� �߾Ӱ��� ���̸� Ȯ�� �� �� ���̰� ������ ok!!
	vector<double> negSlopesGood;
	double negSum = 0.0; // sum so we'll be able to get mean

	// Loop through positive slopes and add the good ones
	for (size_t i = 0; i != negativeSlopes.size(); ++i) {

		// ���� ���簪�� �߾Ӱ����� ���̰� ���� �Ӱ谪���� ���ٸ� ok!
		if (abs(negativeSlopes[i] - negSlopeMedian) < 0.9) { // < negSlopeMedian*0.2
			negSlopesGood.push_back(negativeSlopes[i]); // Add slope to negSlopesGood
			negSum += negativeSlopes[i]; // add to sum
		}
	}

	// ���� ���� ��հ� ���
	double negSlopeMean = negSum / negSlopesGood.size();

	SlopeMean.push_back(posSlopeMean);
	SlopeMean.push_back(negSlopeMean);

	return SlopeMean;

}
vector<double> get_intercept_coord(Mat& image, vector<vector<double>>& slopePositiveLines, vector<vector<double>>& slopeNegativeLines) {
	vector<double> InterceptPos;
	
	// Positive Lines
	vector<double> xInterceptPos; // ������ ��� ����

	// ������ǥ ã��
	for (size_t i = 0; i != slopePositiveLines.size(); ++i) {
		double x1 = slopePositiveLines[i][0]; // x value
		double y1 = image.rows - slopePositiveLines[i][1]; // y�� ����
		double slope = slopePositiveLines[i][4];
		double yIntercept = y1 - slope * x1; // b = y-ax, y����
		double xIntercept = -yIntercept / slope; // y=ax+b=0, x = -b/a , x����
		if (isnan(xIntercept) == 0) { // �߻� ����
			xInterceptPos.push_back(xIntercept);
		}
	}

	// x��ǥ���� �߾Ӱ� ���
	double xIntPosMed = median(xInterceptPos);

	// ���� ���� �����ϱ� ���� �߾Ӱ� ������ �̿��� x��ǥ�� �߾Ӱ� ������ ���� ���� x��ǥ���� ��
	// ���̰� ������ ���� ��!!
	vector<double> xIntPosGood;
	double xIntSum = 0; 

	// �񱳸� ���� ���� �ѹ� ��(���⼱ �߾Ӱ� ���� X)
	for (size_t i = 0; i != slopePositiveLines.size(); ++i) {
		double x1 = slopePositiveLines[i][0]; // x value
		double y1 = image.rows - slopePositiveLines[i][1]; // y�� �����Ǿ� ����
		double slope = slopePositiveLines[i][4];
		double yIntercept = y1 - slope * x1; // b = y-ax, y����
		double xIntercept = -yIntercept / slope; // y=ax+b=0, x = -b/a , x����

		// �߻������� �ʰ� �߾Ӱ��� ũ�� ���̰� ���� �ʴ´ٸ� ���
		if (isnan(xIntercept) == 0 && abs(xIntercept - xIntPosMed) < 0.35*xIntPosMed) {
			xIntPosGood.push_back(xIntercept); // add to 'good' vector
			xIntSum += xIntercept;
		}
	}

	// ���� ������ x��ǥ ��հ� ���
	double xInterceptPosMean = xIntSum / xIntPosGood.size();

	/*---------------------------------------------------------------------*/

	// Negative Lines
	vector<double> xInterceptNeg; //������ ��� ����

	// ������ǥ ã��
	for (size_t i = 0; i != slopeNegativeLines.size(); ++i) {
		double x1 = slopeNegativeLines[i][0]; // x value
		double y1 = image.rows - slopeNegativeLines[i][1]; // y�� ����
		double slope = slopeNegativeLines[i][4];
		double yIntercept = y1 - slope * x1; // b = y-ax, y����
		double xIntercept = -yIntercept / slope; // y=ax+b=0, x = -b/a , x����
		if (isnan(xIntercept) == 0) { // �߻� ����
			xInterceptNeg.push_back(xIntercept); 
		}
	}

	// x��ǥ���� �߾Ӱ� ���
	double xIntNegMed = median(xInterceptNeg);

	// ���� ���� �����ϱ� ���� �߾Ӱ� ������ �̿��� x��ǥ�� �߾Ӱ� ������ ���� ���� x��ǥ���� ��
	// ���̰� ������ ���� ��!!
	vector<double> xIntNegGood;
	double xIntSumNeg = 0; 

	// �񱳸� ���� ���� �ѹ� ��(���⼱ �߾Ӱ� ���� X)
	for (size_t i = 0; i != slopeNegativeLines.size(); ++i) {
		double x1 = slopeNegativeLines[i][0]; // x value
		double y1 = image.rows - slopeNegativeLines[i][1]; // y�����
		double slope = slopeNegativeLines[i][4];
		double yIntercept = y1 - slope * x1; // b = y-ax, y����
		double xIntercept = -yIntercept / slope; // y=ax+b=0, x = -b/a , x����

		// �߻������� �ʰ� �߾Ӱ��� ũ�� ���̰� ���� �ʴ´ٸ� ���
		if (isnan(xIntercept) == 0 && abs(xIntercept - xIntNegMed) < .35*xIntNegMed) {
			xIntNegGood.push_back(xIntercept); 
			xIntSumNeg += xIntercept;
		}
	}

	// ���� ������ x��ǥ ��հ� ���
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
	double y2 = image.size().height - (image.size().height - image.size().height*0.35); // ROI��ٸ��� ����
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

	//���� ���� ���� 
	Scalar lower_white = Scalar(200, 200, 200); //��� ���� (RGB)
	Scalar upper_white = Scalar(255, 255, 255);
	Scalar lower_yellow = Scalar(10, 100, 100); //����� ���� (HSV)
	Scalar upper_yellow = Scalar(40, 255, 255);

	//Filter white pixels
	inRange(img_bgr, lower_white, upper_white, white_mask); // ��� ������
	bitwise_and(img_bgr, img_bgr, white_image, white_mask); // input �̹������� �ش� ��谪 ���� �ִ� ���� ����

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
	//fillter_colors(image, img_filtered); // �� �����ϱ� ���� ��� HSV�� ��ȯ����
	
	Mat gray = gray_scale(image); // gray scale
	Mat blur = Gaussian_blurring(gray); // gaussian filtering
	Mat edge = canny_edge(blur); // canny edge detection
	Mat roi = Roi(image, edge); // roi
	vector<Vec4i> lines = hough_transform(image, roi); // hoguh transform
	
	
	// �̶� �׾��� ���� ���� �߿� ����ȭ�� ������ �����ϱ� ���� 
	// ���ε��� �߾Ӱ��� ������ �ض�� �����ϰ� �߾� ������ ��ǥ�� ������!!
	// ���ؾ� �� �� : x,y��ǥ �߾Ӱ� ���� �߾Ӱ�

	// ���� ���ΰ� ���� ������ ������ ���� ���⸦ �������� ������.
	vector< vector<double> > slopePositiveLines; // [x1 y1 x2 y2 slope]
	vector< vector<double> > slopeNegativeLines; // [x1 y1 x2 y2 slope]
	seperate_line(lines, slopePositiveLines, slopeNegativeLines);

	// �������� ���͵��� ���� �߾Ӱ��� ���Ѵ�.
	vector<double> SlopeMean = get_slope_mean(slopePositiveLines, slopeNegativeLines); 
	double posSlopeMean = SlopeMean[0]; // first = positive
	double negSlopeMean = SlopeMean[1]; // second = negative

	// �������� ���͵��� ��ǥ �߾Ӱ� ���ϱ�
	vector<double> InterceptPos = get_intercept_coord(image, slopePositiveLines, slopeNegativeLines);
	double xInterceptPosMean = InterceptPos[0]; // first = positive
	double xInterceptNegMean = InterceptPos[1]; // second = negative

	// ��� y���� ��ǥ�� ���� ������ roi��ٸ����� ���̰� �� y��ǥ�̱� ����
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
		fillter_colors(image, img_filtered); // �� �����ϱ� ���� ��� HSV�� ��ȯ����

		Mat gray = gray_scale(img_filtered); // gray scale
		Mat blur = Gaussian_blurring(gray); // gaussian filtering
		Mat edge = canny_edge(blur); // canny edge detection
		Mat roi = Roi(image, edge); // roi
		vector<Vec4i> lines = hough_transform(image, roi); // hoguh transform


		// �̶� �׾��� ���� ���� �߿� ����ȭ�� ������ �����ϱ� ���� 
		// ���ε��� �߾Ӱ��� ������ �ض�� �����ϰ� �߾� ������ ��ǥ�� ������!!
		// ���ؾ� �� �� : x,y��ǥ �߾Ӱ� ���� �߾Ӱ�

		// ���� ���ΰ� ���� ������ ������ ���� ���⸦ �������� ������.
		vector< vector<double> > slopePositiveLines; // [x1 y1 x2 y2 slope]
		vector< vector<double> > slopeNegativeLines; // [x1 y1 x2 y2 slope]
		seperate_line(lines, slopePositiveLines, slopeNegativeLines);

		// �������� ���͵��� ���� �߾Ӱ��� ���Ѵ�.
		vector<double> SlopeMean = get_slope_mean(slopePositiveLines, slopeNegativeLines);
		double posSlopeMean = SlopeMean[0]; // first = positive
		double negSlopeMean = SlopeMean[1]; // second = negative

		// �������� ���͵��� ��ǥ �߾Ӱ� ���ϱ�
		vector<double> InterceptPos = get_intercept_coord(image, slopePositiveLines, slopeNegativeLines);
		double xInterceptPosMean = InterceptPos[0]; // first = positive
		double xInterceptNegMean = InterceptPos[1]; // second = negative

		// ������ ����� drawing_lane return void���� Mat����
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