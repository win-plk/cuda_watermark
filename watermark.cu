#include<stdio.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<iostream>

#define T 256 //number of threads
#define TRANSPARENCY 80 // level of transparency

using namespace cv;
using namespace std;

__global__ void addwatermark(unsigned char *wpp, unsigned char *wtm, int h_wpp, int w_wpp, int h_wtm, int w_wtm){
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	
	int Index_wtm = yIndex * w_wtm + xIndex;
	int Index_wpp = (h_wpp - h_wtm + yIndex) * w_wpp + xIndex;
	
	if((xIndex<w_wtm)&&(yIndex<h_wtm)){
		wpp[Index_wpp*3] = ((wpp[Index_wpp*3]*TRANSPARENCY)+(wtm[Index_wtm*3]*(100-TRANSPARENCY)))*0.01;
		wpp[Index_wpp*3+1] = ((wpp[Index_wpp*3+1]*TRANSPARENCY)+(wtm[Index_wtm*3+1]*(100-TRANSPARENCY)))*0.01;
		wpp[Index_wpp*3+2] = ((wpp[Index_wpp*3+2]*TRANSPARENCY)+(wtm[Index_wtm*3+2]*(100-TRANSPARENCY)))*0.01;
	}
}

int main(){
	Mat img_wallpaper = imread("pic\\wallpaper.jpg", IMREAD_COLOR);
	Mat img_water = imread("pic\\water.jpg", IMREAD_COLOR);
	
	imshow("Original", img_wallpaper);
	
	cout << "wpp" << img_wallpaper.rows <<" x "<< img_wallpaper.cols<<endl<<"wtm" << img_water.rows <<" x "<< img_water.cols<<endl;
	
	unsigned char *input_wpp = (unsigned char*)(img_wallpaper.data);
	unsigned char *input_wtm = (unsigned char*)(img_water.data);
	
	int size_wallpaper = sizeof(char) * 3 * img_wallpaper.rows * img_wallpaper.cols;
	int size_water = sizeof(char) * 3 * img_water.rows * img_water.cols;
	
	unsigned char *dev_wpp, *dev_wtm;
	
	cudaMalloc( (void**)&dev_wpp, size_wallpaper);
	cudaMalloc( (void**)&dev_wtm, size_water);
	
	cudaMemcpy( dev_wpp, input_wpp, size_wallpaper, cudaMemcpyHostToDevice);
	cudaMemcpy( dev_wtm, input_wtm, size_water, cudaMemcpyHostToDevice);
	
	dim3 dimblock(16, 16);
	dim3 dimgrid((img_water.cols + dimblock.x - 1)/dimblock.x, (img_water.rows + dimblock.y - 1)/dimblock.y);
	
	addwatermark<<<dimgrid, dimblock>>>(dev_wpp, dev_wtm, img_wallpaper.rows, img_wallpaper.cols, img_water.rows, img_water.cols);
	
	cudaMemcpy( input_wpp, dev_wpp, size_wallpaper, cudaMemcpyDeviceToHost);
	
	Mat img_output =  Mat(img_wallpaper.rows, img_wallpaper.cols, CV_8UC3, input_wpp);
	
	cudaFree(dev_wpp);
	cudaFree(dev_wtm);
	
	imwrite("\\output.jpg", img_output);
	imshow("Modified", img_output);
	
	waitKey();
	
	return 0;
}