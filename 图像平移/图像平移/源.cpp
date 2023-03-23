//ͼ��任
//by sundule
#include <opencv2\opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>
#include<vector>
#define PI 3.1415926535
using namespace cv;
using namespace std;

//i   dst��������
//j   dst��������
//u    j����ӳ�䵽src�е�������
//v    i����ӳ�䵽src�е�������

//˫���Բ�ֵ
void BilinearInterpolation(Mat& src, Mat& dst, int i, int j, double u, double v)
{
	if (u >= 0 && v >= 0 && u <= src.cols - 1 && v <= src.rows - 1)
	{
		int x1 = floor(u);
		int x2 = ceil(u);
		int y1 = floor(v);
		int y2 = ceil(v);
		double pu = u - x1; //�е�С������
		double pv = v - y1; //�е�С������
		
		dst.at<uchar>(i, j) = (1 - pv) * (1 - pu) * src.at<uchar>(y1, x1) +
			(1 - pv) * pu * src.at<uchar>(y1, x2) +
			pv * (1 - pu) * src.at<uchar>(y2, x1) + pv * pu * src.at<uchar>(y2, x2);
	}
}

void translateTransform(Mat& src, Mat& dst, double tx, double ty)
{
	//�������ͼ��
	int dst_H = src.rows;
	int dst_W = src.cols;
	dst = Mat::zeros(dst_H, dst_W, CV_8UC1); 
	//�Ҷ�ͼ���ʼ��
	Mat T = (Mat_<double>(3, 3) <<	1, 0, tx,
									0, 1, ty, 
									0, 0, 1); 
	Mat T_inv = T.inv();
	//����ӳ��
	for (int i = 0; i < dst_H; i++)
	{
		for (int j = 0; j < dst_W; j++)
		{
			Mat dst_uv = (Mat_<double>(3, 1) << j, i, 1);
			Mat src_uv = T_inv * dst_uv;
			double u = src_uv.at<double>(0, 0); //ԭͼ��ĺ����꣬�У���
			double v = src_uv.at<double>(1, 0); //ԭͼ��������꣬�У���
			//double w = src_uv.at<double>(2, 0);
			BilinearInterpolation(src, dst, i, j, u, v);
		}
	}
}


void scaleTransform(Mat& src, Mat& dst, double sx, double sy) 
{
	int dst_H = round(sy * src.rows);
	int dst_W = round(sx * src.cols);
	dst = Mat::zeros(dst_H, dst_W, CV_8UC1); 
	Mat T = (Mat_<double>(3, 3) << sx, 0, 0, 
									0, sy, 0, 
									0, 0, 1); 
	Mat T_inv = T.inv(); 
	for (int i = 0; i < dst_H; i++)
	{
		for (int j = 0; j < dst_W; j++)
		{
			Mat dst_uv = (Mat_<double>(3, 1) << j, i, 1);
			Mat src_uv = T_inv * dst_uv;
			double u = src_uv.at<double>(0, 0); 
			double v = src_uv.at<double>(1, 0);
			//double w = src_uv.at<double>(2, 0);
			if (u < 0) u = 0;
			if (v < 0) v = 0;
			if (u > src.cols - 1) u = src.cols - 1;
			if (v > src.rows - 1) v = src.rows - 1;
			BilinearInterpolation(src, dst, i, j, u, v);
		}
	}
}


void mirrorTransform(Mat& src, Mat& dst)
{
	int dst_H = src.rows;
	int dst_W = src.cols;
	dst = Mat::zeros(dst_H, dst_W, CV_8UC1);

	Mat T1 = (Mat_<double>(3, 3) << -1, 0, 0,
									0, 1, 0,
									0, 0, 1);
	Mat T2 = (Mat_<double>(3, 3) << 1, 0, src.cols,
									0, 1, 0,
									0, 0, 1);//�任����x�����ᣬ�����������ƽ��
	T1.at<double>(0, 0) = 1;
	T1.at<double>(1, 1) = -1;
	T2.at<double>(0, 2) = 0;
	T2.at<double>(1, 2) = src.rows;
	Mat T = T2 * T1;
	Mat T_inv = T.inv();

	for (int i = 0; i < dst_H; i++)
	{
		for (int j = 0; j < dst_W; j++)
		{
			Mat dst_uv = (Mat_<double>(3, 1) << j, i, 1);
			Mat src_uv = T_inv * dst_uv;
			double u = src_uv.at<double>(0, 0);
			double v = src_uv.at<double>(1, 0);
			//double w = src_uv.at<double>(2, 0);

			if (u < 0) u = 0;
			if (v < 0) v = 0;
			if (u > src.cols - 1) u = src.cols - 1;
			if (v > src.rows - 1) v = src.rows - 1;

			BilinearInterpolation(src, dst, i, j, u, v);
		}
	}
}


void deviationTransform(Mat& src, Mat& dst, double dx, double dy)
{
	int dst_H = fabs(dy) * src.cols + src.rows;
	int dst_W = fabs(dx) * src.rows + src.cols;
	dst = Mat::zeros(dst_H, dst_W, CV_8UC1);
	// ��ԭͼ������ӳ�䵽��ѧ�ѿ�������
	Mat T1 = (Mat_<double>(3, 3) << 1, 0, -0.5 * src.cols, 
									0, -1, 0.5 * src.rows, 
									0, 0, 1); 
	// ��ѧ�ѿ�������ƫ�Ʊ任����
	Mat T2 = (Mat_<double>(3, 3) << 1, dx, 0, 
									dy, 1, 0, 
									0, 0, 1); 
	// ����ѧ�ѿ�������ӳ�䵽��ת���ͼ������
	Mat T3 = (Mat_<double>(3, 3) <<1, 0, 0.5 * dst.cols,
									0, -1, 0.5 * dst.rows, 
									0, 0, 1 ); 

	Mat T = T3 * T2 * T1;
	Mat T_inv = T.inv(); 
	for (int i = 0; i < dst_H; i++) 
	{
		for (int j = 0; j < dst_W; j++) 
		{
			Mat dst_uv = (Mat_<double>(3, 1) << j, i, 1);
			Mat src_uv = T_inv * dst_uv;
			double u = src_uv.at<double>(0, 0);
			double v = src_uv.at<double>(1, 0);
			//double w = src_uv.at<double>(2, 0);

			BilinearInterpolation(src, dst, i, j, u, v);			
		}
	}
}


void rotateTransform(Mat& src, Mat& dst, double Angle) 
{
	double angle = Angle * PI / 180.0;
	int dst_H = round(fabs(src.rows * cos(angle)) + fabs(src.cols * sin(angle)));
	int dst_W = round(fabs(src.cols * cos(angle)) + fabs(src.rows * sin(angle)));
	dst = Mat::zeros(dst_H, dst_W, CV_8UC1); 
	Mat T1 = (Mat_<double>(3, 3) << 1, 0, -0.5 * src.cols, 
									0, -1, 0.5 * src.rows, 
									0, 0, 1); 
	Mat T2 = (Mat_<double>(3, 3) << cos(angle), sin(angle), 0, 
									-sin(angle), cos(angle), 0, 
									0, 0, 1); 
	Mat T3 = (Mat_<double>(3, 3) << 1, 0, 0.5 * dst.cols, 
									0, -1, 0.5 * dst.rows, 
									0, 0, 1); 
	Mat T = T3 * T2 * T1;
	//cout << T << endl;
	Mat T_inv = T.inv();

	for (int i = 0; i < dst.rows; i++) 
	{
		for (int j = 0; j < dst.cols; j++) 
		{
			Mat dst_uv = (Mat_<double>(3, 1) << j, i, 1);
			Mat src_uv = T_inv * dst_uv;
			double u = src_uv.at<double>(0, 0);
			double v = src_uv.at<double>(1, 0);
			//double w = src_uv.at<double>(2, 0);
			/*if (int(Angle) % 90 == 0)
			{
				if (u < 0) u = 0; 
				if (u > src.cols - 1) u = src.cols - 1;//�����
				if (v < 0) v = 0; 
				if (v > src.rows - 1) v = src.rows - 1; 
			}*/
			BilinearInterpolation(src, dst, i, j, u, v);
		}
	}
}

/*
void perspectiveTransform(Mat& src1, Mat& dst)
{
	int dst_H = src1.rows;
	int dst_W = src1.cols;
	dst = Mat::zeros(dst_H, dst_W, CV_8UC3);

	vector<Point2f> corner(4);
	//Point2f corner[4];
	corner[0] = Point2f(0, 0);
	corner[1] = Point2f(src1.cols - 1, 0);
	corner[2] = Point2f(0, src1.rows - 1);
	corner[3] = Point2f(src1.cols - 1, src1.rows - 1);
	vector<Point2f> corner_trans(4);
	//Point2f corner_trans[4];
	corner_trans[0] = Point2f(dst_W * 0.2, dst_H * 0.2);
	corner_trans[1] = Point2f(dst_W * 0.7, 0);
	corner_trans[2] = Point2f(0, dst_H * 0.9);
	corner_trans[3] = Point2f(dst_W * 0.6, dst_H * 0.6);
	//vector<float> point;
	//vector<float> point_trans;
	vector<Point2f> point, point_trans;


	for (int i = 0; i < dst_H; i++) 
	{
		for (int j = 0; j < dst_W; j++) 
		{
			point.push_back(Point2f(j, i));
		}
	}

	Mat transform = getPerspectiveTransform(corner, corner_trans);
	perspectiveTransform(point, point_trans, transform);
	int count = 0;
	for (int i = 0; i < dst_H; i++) {
		uchar* p = dst.ptr<uchar>(i);
		for (int j = 0; j < dst_W; j++) {
			int y = point_trans[count].y;
			int x = point_trans[count].x;
			uchar* t = dst.ptr<uchar>(y);
			t[x * 3] = p[j * 3];
			t[x * 3 + 1] = p[j * 3 + 1];
			t[x * 3 + 2] = p[j * 3 + 2];
			count++;
		}
	}


	vector<float> point;
	vector<float> point_trans;
	//vector<Point2f> point;
	//vector<Point2f> point_trans;

	for (int i = 0; i < dst_H; i++)
	{
		for (int j = 0; j < dst_W; j++)
		{
			//point.push_back(Point2f(j, i));
			point.push_back(j);//β����������
			point.push_back(i);
		}
	}

	for (int i = 0; i < dst_H; i++)
	{
		uchar* t = dst.ptr<uchar>(i);
		for (int j = 0; j < dst_W; j++)
		{
			int tmp = i * dst_W + j;
			 int x = point[tmp * 2];
			 int y = point[tmp * 2 + 1];
			//int y = point_trans[count].y;
			//int x = point_trans[count].x;
			//uchar* t = dst.ptr<uchar>(y);
			uchar* p = dst.ptr<uchar>(y);
			t[x * 3] = p[j * 3];
			t[x * 3 + 1] = p[j * 3 + 1];
			t[x * 3 + 2] = p[j * 3 + 2];
		}
	}

}*/


int main() 
{
	Mat src = imread("D:\dog.jpg",0);
	Mat src1 = imread("D:\dog.jpg", 1);
	Mat dst, dst1, dst2, dst3, dst4;
	//Mat dst5;
	//double w = 1;

	double angle = 90; 
	double tx = 50, ty = 50; 
	double sx = 1.5, sy = 1; 
	double dx = 0.2, dy = 0.2; 

	rotateTransform(src, dst, angle); 
	translateTransform(src, dst1, tx, ty); 
	scaleTransform(src, dst2, sx, sy);  
	deviationTransform(src, dst3, dx, dy); 
	mirrorTransform(src, dst4);
	//perspectiveTransform(src, dst5);

	imwrite("result.jpg", dst);
	namedWindow("ԭͼ");
	imshow("ԭͼ", src);
	namedWindow("��ת");
	imshow("��ת", dst);
	namedWindow("ƽ��");
	imshow("ƽ��", dst1);
	namedWindow("����");
	imshow("����", dst2);
	namedWindow("����");
	imshow("����", dst3);
	namedWindow("����");
	imshow("����", dst4);
	//namedWindow("ͶӰ");
	//imshow("ͶӰ", dst5);
	//cout << dst.size << endl;
	
	waitKey(0);

	return 0;
}