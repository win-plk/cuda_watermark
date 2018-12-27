#include<stdio.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<iostream>

#define TRANSPARENCY 80 // level of transparency

using namespace cv;
using namespace std;

__global__ void addwatermark(unsigned char *wpp, unsigned char *wtm, int h_wpp, int w_wpp, int h_wtm, int w_wtm){
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	
	int Index_wtm = yIndex * w_wtm + xIndex;
	int Index_wpp = (h_wpp - h_wtm + yIndex) * w_wpp + xIndex;
	
	if((xIndex<w_wtm)&&(yIndex<h_wtm)){
		if(wtm[Index_wtm*3]==0 && wtm[Index_wtm*3+1]==0 && wtm[Index_wtm*3+2]==0){
		}else{
		wpp[Index_wpp*3] = ((wpp[Index_wpp*3]*TRANSPARENCY)+(wtm[Index_wtm*3]*(100-TRANSPARENCY)))*0.01;
		wpp[Index_wpp*3+1] = ((wpp[Index_wpp*3+1]*TRANSPARENCY)+(wtm[Index_wtm*3+1]*(100-TRANSPARENCY)))*0.01;
		wpp[Index_wpp*3+2] = ((wpp[Index_wpp*3+2]*TRANSPARENCY)+(wtm[Index_wtm*3+2]*(100-TRANSPARENCY)))*0.01;
		}
	}
}

int main(int argc, char* argv[]){
	// Load Images 
	Mat img_wallpaper = imread(argv[1], IMREAD_COLOR);
	Mat img_water = imread(argv[2], IMREAD_COLOR);
    if(img_wallpaper.empty() || img_water.empty()){
        cout <<  "Could not load the image" << endl ;
        return -1;
    }
	if(img_wallpaper.rows < img_water.rows || img_wallpaper.cols < img_water.cols ){
		cout <<  "Size of watermark is bigger than wallpaper" << endl;
        return -1;
	}
	
	// Show Original Image
	imshow("Original", img_wallpaper);
	
	// Convert Datatype (Mat --> unsigned char) 
	unsigned char *input_wpp = (unsigned char*)(img_wallpaper.data);
	unsigned char *input_wtm = (unsigned char*)(img_water.data);
	
	// Allocate Global Memory Space in GPU
	int size_wallpaper = sizeof(char) * 3 * img_wallpaper.rows * img_wallpaper.cols;
	int size_water = sizeof(char) * 3 * img_water.rows * img_water.cols;
	
	unsigned char *dev_wpp, *dev_wtm;
	
	cudaMalloc( (void**)&dev_wpp, size_wallpaper);
	cudaMalloc( (void**)&dev_wtm, size_water);
	
	// Copy Data From CPU -> GPU
	cudaMemcpy( dev_wpp, input_wpp, size_wallpaper, cudaMemcpyHostToDevice);
	cudaMemcpy( dev_wtm, input_wtm, size_water, cudaMemcpyHostToDevice);
	
	// Set number of thread and block
	dim3 dimblock(16, 16);
	dim3 dimgrid((img_water.cols + dimblock.x - 1)/dimblock.x, (img_water.rows + dimblock.y - 1)/dimblock.y);
	
	// Call Kernel Routine
	addwatermark<<<dimgrid, dimblock>>>(dev_wpp, dev_wtm, img_wallpaper.rows, img_wallpaper.cols, img_water.rows, img_water.cols);
	
	// Copy Data Back From GPU -> CPU
	cudaMemcpy( input_wpp, dev_wpp, size_wallpaper, cudaMemcpyDeviceToHost);
	
	// Convert Datatype Back(unsigned char --> Mat) 
	Mat img_output =  Mat(img_wallpaper.rows, img_wallpaper.cols, CV_8UC3, input_wpp);
	
	// Free Memory Space in GPU
	cudaFree(dev_wpp);
	cudaFree(dev_wtm);
	
	// Write & Show Image
	imwrite("output.jpg", img_output);
	imshow("Modified", img_output);
	
	waitKey();
	
	return 0;
}