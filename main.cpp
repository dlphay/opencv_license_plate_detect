// opencv��ص�ͷ�ļ�
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\types_c.h> 
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>

 
#include <highgui.h>  

using namespace cv;
int main()
{
    //��ȡͼ��Դ
    cv::Mat srcImage = cv::imread("demo0.jpg");  // ͼƬ·�� 

	//���ͼƬ��ʧ�����˳�
    if (srcImage.empty()) 
	{
        return -1;  // �˳�
    }
    //תΪ�Ҷ�ͼ��
    cv::Mat srcGray;//�����޳�ʼ������
    cv::cvtColor(srcImage, srcGray,CV_RGB2GRAY);// ͼ��ĻҶ�ת��

	Mat srcGray_output = srcGray.clone();  

	for(int x = 0;x<srcGray_output.rows;x++)       
    {       
        for(int y = 0;y<srcGray_output.cols;y++)  
        {  
			//if(x < 138 || x>480)   srcGray_output.at<uchar>(x,y) = 0;
			//if(y < 200 || y>360)   srcGray_output.at<uchar>(x,y) = 0;
        }  
    }  
	threshold(srcGray, srcGray, 230, 255, CV_THRESH_BINARY);  //ͼ���ֵ��
	Mat erzhihua_IMAGE = srcGray.clone(); // ��ֵ�� 
	//��̬ѧ�Ĵ���
	//������ (ȥ��һЩ���)  �����ֵ����ͼƬ���Ų�����Ȼ�ܶ࣬���������size  
	Mat element = getStructuringElement(MORPH_RECT, Size(1,1));
	morphologyEx(srcGray, srcGray, MORPH_OPEN, element);   
	morphologyEx(srcGray, srcGray, MORPH_CLOSE, element);

	Mat AAAAA_IMAGE = srcGray.clone(); // ����ͼƬ

	// ͼ������б�����ȥ����
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
	//Sobel��Ե����
	// X�����ı�Ե����
	Sobel(srcGray_output,OUTPUT_SOBEL_x,srcGray_output.depth(),1,0,1,1,0,BORDER_DEFAULT ); 
	// Y�����ı�Ե����
	Sobel(srcGray_output,OUTPUT_SOBEL_y,srcGray_output.depth(),0,1,1,1,0,BORDER_DEFAULT ); 
	Mat OUTPUT_SOBEL = srcGray_output.clone(); 

	float alpha = 1;  // x�� ����ϵ��
	float beta = 1;   // y�� ����ϵ��
	addWeighted( OUTPUT_SOBEL_x, alpha, OUTPUT_SOBEL_y, beta, 0.0, OUTPUT_SOBEL); //

	//OUTPUT_SOBEL=cvCreateImage(cvGetSize(srcGray_output),IPL_DEPTH_16S,1);  
	//Sobel(srcGray_output,OUTPUT_SOBEL,1,0,3);


    cv::imshow("ԭͼ��", srcImage);//��ʾԴͼ��
    cv::imshow("�ҶȻ�", srcGray_output);//��ʾ�Ҷ�ͼ��
	cv::imshow("��ֵ��", erzhihua_IMAGE);//��ʾ��ֵ��
	cv::imshow("Sobel_x��Ե", OUTPUT_SOBEL_x);//��ʾ��̬ѧ
	cv::imshow("Sobel_y��Ե", OUTPUT_SOBEL_y);//��ʾ��̬ѧ
	cv::imshow("Sobel x������y�������Ե���", OUTPUT_SOBEL);//��ʾ��̬ѧ
	cv::imshow("���ͼ", OUTPUT_IMAGE);//��ʾ���ͼ
	cv::imshow("���ս��ͼ", AAAAA_IMAGE);//��ʾ����ͼ

    cv::waitKey(0); //�ȴ���������ͼƬһֱ��ʾ�⡣
    return 0;
}