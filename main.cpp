// opencv相关的头文件
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\types_c.h> 
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>

 
#include <highgui.h>  

using namespace cv;
int main()
{
    //读取图像源
    cv::Mat srcImage = cv::imread("demo0.jpg");  // 图片路径 

	//如果图片打开失败则退出
    if (srcImage.empty()) 
	{
        return -1;  // 退出
    }
    //转为灰度图像
    cv::Mat srcGray;//创建无初始化矩阵
    cv::cvtColor(srcImage, srcGray,CV_RGB2GRAY);// 图像的灰度转换

	Mat srcGray_output = srcGray.clone();  

	for(int x = 0;x<srcGray_output.rows;x++)       
    {       
        for(int y = 0;y<srcGray_output.cols;y++)  
        {  
			//if(x < 138 || x>480)   srcGray_output.at<uchar>(x,y) = 0;
			//if(y < 200 || y>360)   srcGray_output.at<uchar>(x,y) = 0;
        }  
    }  
	threshold(srcGray, srcGray, 230, 255, CV_THRESH_BINARY);  //图像二值化
	Mat erzhihua_IMAGE = srcGray.clone(); // 二值化 
	//形态学的处理
	//开操作 (去除一些噪点)  如果二值化后图片干扰部分依然很多，增大下面的size  
	Mat element = getStructuringElement(MORPH_RECT, Size(1,1));
	morphologyEx(srcGray, srcGray, MORPH_OPEN, element);   
	morphologyEx(srcGray, srcGray, MORPH_CLOSE, element);

	Mat AAAAA_IMAGE = srcGray.clone(); // 最终图片

	// 图像的行列遍历，去噪声
	for(int x = 0;x<srcGray.rows;x++)       
    {       
        for(int y = 0;y<srcGray.cols;y++)  
        {  
			if(x < 200 || x>280)   srcGray.at<uchar>(x,y) = 0;
			if(y < 210 || y>380)   srcGray.at<uchar>(x,y) = 0;
        }  
    }  
	AAAAA_IMAGE = AAAAA_IMAGE(Range(210, 260), Range(210, 380));

	Mat OUTPUT_IMAGE = srcGray.clone(); 
	Mat OUTPUT_SOBEL_x = srcGray_output.clone(); 
	Mat OUTPUT_SOBEL_y = srcGray_output.clone(); 
	//Sobel边缘处理
	// X分量的边缘处理
	Sobel(srcGray_output,OUTPUT_SOBEL_x,srcGray_output.depth(),1,0,1,1,0,BORDER_DEFAULT ); 
	// Y分量的边缘处理
	Sobel(srcGray_output,OUTPUT_SOBEL_y,srcGray_output.depth(),0,1,1,1,0,BORDER_DEFAULT ); 
	Mat OUTPUT_SOBEL = srcGray_output.clone(); 

	float alpha = 1;  // x的 叠加系数
	float beta = 1;   // y的 叠加系数
	addWeighted( OUTPUT_SOBEL_x, alpha, OUTPUT_SOBEL_y, beta, 0.0, OUTPUT_SOBEL); //

	//OUTPUT_SOBEL=cvCreateImage(cvGetSize(srcGray_output),IPL_DEPTH_16S,1);  
	//Sobel(srcGray_output,OUTPUT_SOBEL,1,0,3);


    cv::imshow("原图像", srcImage);//显示源图像
    cv::imshow("灰度化", srcGray_output);//显示灰度图像
	cv::imshow("二值化", erzhihua_IMAGE);//显示二值化
	cv::imshow("Sobel_x边缘", OUTPUT_SOBEL_x);//显示形态学
	cv::imshow("Sobel_y边缘", OUTPUT_SOBEL_y);//显示形态学
	cv::imshow("Sobel x分量与y分量线性叠加", OUTPUT_SOBEL);//显示形态学
	cv::imshow("结果图", OUTPUT_IMAGE);//显示结果图
	cv::imshow("最终结果图", AAAAA_IMAGE);//显示最终图

    cv::waitKey(0); //等待。可以让图片一直显示这。
    return 0;
}