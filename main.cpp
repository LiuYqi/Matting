#include "sharedmatting.h"
#include <string>
#include <opencv2/opencv.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
using namespace cv;
using namespace std;      
int g_nThresholdValue = 100;
int g_nThresholdType = 3;
Mat g_srcImage, g_grayImage, g_dstImage;
using namespace std;

int main()
{
	Mat src = imread("input/lady1.jpg");
	Mat out, can1,can2, can, ero, dil,trimap;
	boxFilter(src, out, -1, Size(10, 10));
	Canny(out, can1, 36, 36, 3);
       
	Rect ccomp;
	floodFill(out, Point(10, 10), Scalar(0, 0, 0), &ccomp, Scalar(3, 3, 3), Scalar(3, 3, 3));
	Canny(out, can2, 15, 15, 3);
	addWeighted(can1, 1.0, can2, 1.0, 0.0, can);

	/*floodFill(out, Point(10, 10), Scalar(0, 0, 0), &ccomp, Scalar(3, 3, 3), Scalar(3, 3, 3));
	floodFill(out, Point(1963, 2400), Scalar(0, 0, 0), &ccomp, Scalar(3, 3, 3), Scalar(3, 3, 3));*/
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(35, 35));
	Mat element_e = getStructuringElement(MORPH_ELLIPSE, Size(50, 50));
	dilate(can, dil, element_e);
	dilate(dil, dil, element_e);
	dilate(dil, dil, element);
	dilate(dil, dil, element);
	erode(dil, ero, element_e);
	erode(ero, ero, element_e);
	erode(ero, ero, element_e);
	erode(ero, ero, element_e);
	erode(ero, ero, element);
    erode(ero, ero, element);
	//cvtColor(out, g_grayImage, CV_RGB2GRAY);
	//threshold(g_grayImage, g_dstImage, g_nThresholdValue, 255, g_nThresholdType);
	addWeighted(dil,0.5,ero,0.5,0.0,trimap);
	//imwrite("dil.jpg", dil);
	//imwrite("ero.jpg", ero);
	imwrite("result/lady_trimap1.jpg", trimap);
	cout << "ok" << endl; 

        SharedMatting sm;
        sm.loadImage("input/lady1.jpg");
        sm.loadTrimap("result/lady_trimap1.jpg");
        sm.solveAlpha();
        sm.save("result/lady_alpha1.jpg");
        int ** alpha = sm.alpha_return();
        Mat b_k = imread("background/background.jpg");
        Mat alp = imread("result/lady_alpha1.jpg");
        boxFilter(alp, alp, -1, Size(10, 10));
        imwrite("result/lady_alpha1.jpg", alp);
        int src_row = src.rows;
        int src_col = src.cols;
        int i,j;
       for (i=0; i<src_row ;i++) 
        {
          
           for (j=0; j<src_col; j++)
           {
              alpha[i][j] = double (alp.at<Vec3b>(i,j)[0]); 
           }
        }
    
        resize(b_k, b_k, src.size());

        for (i=0; i<src_row ;i++) 
        {
           uchar* data_s = src.ptr<uchar>(i);
           uchar* data_b = b_k.ptr<uchar>(i);
           uchar* data_a = alp.ptr<uchar>(i);
           for (j=0; j<src_col; j++)
           {
              data_s[3*j] = uchar(double(data_s[3*j]) * double(alpha[i][j])/255.0 + double(data_b[3*j]) * (1.0 - double(alpha[i][j])/255.0)); 
             data_s[3*j+1] = uchar(double(data_s[3*j+1]) * double(alpha[i][j])/255.0 + double(data_b[3*j+1]) * (1.0 - double(alpha[i][j])/255.0)); 
             data_s[3*j+2] = uchar(double(data_s[3*j+2]) * double(alpha[i][j])/255.0 + double(data_b[3*j+2]) * (1.0 - double(alpha[i][j])/255.0)); 
           }
        }
        imwrite("result/lady_result1.jpg", src);
        resize(src, src, src.size());
        imshow("result",src);
        waitKey(0);
    return 0;
}
