#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <ctime>
#include <time.h>
#include <chrono>

using namespace std;
using namespace cv;

#define TIC \
    auto t_start = std::chrono::high_resolution_clock::now();
#define TOC                                                 \
    auto t_end = std::chrono::high_resolution_clock::now(); \
    float inf_time1 = std::chrono::duration<float, std::milli>(t_end - t_start).count();

cv::Mat cuda_hist_proc(cv::Mat& input){
    cv::Mat RGB[3];
    cv::split(input, RGB);
    cv::cuda::GpuMat gpuRGB[3];
    TIC
    for (int i=0; i < 3; i++){
        gpuRGB[i].upload(RGB[i]);

        cv::cuda::equalizeHist(gpuRGB[i], gpuRGB[i]);

        gpuRGB[i].download(RGB[i]);
    }
    TOC
    cout<<"cuda time: "<<inf_time1<<endl;
    cv::Mat out;
    merge(RGB, 3, out);
    return out;
}

cv::Mat cpu_hist_proc(cv::Mat& input){
    cv::Mat RGB[3];
    split(input,RGB);
    TIC
    for (int i=0; i < 3; i++){
        cv::equalizeHist(RGB[i], RGB[i]);
    }
    TOC
    cout<<"cpu time: "<<inf_time1<<endl;
    cv::Mat out;
    merge(RGB, 3, out);
    return out;
}


int main() {
    int num_devices = cv::cuda::getCudaEnabledDeviceCount();

    if (num_devices <= 0) {
        std::cerr << "There is no device." << std::endl;
        return -1;
    }
    int enable_device_id = -1;
    for (int i = 0; i < num_devices; i++) {
        cv::cuda::DeviceInfo dev_info(i);
        if (dev_info.isCompatible()) {
            enable_device_id = i;
        }
    }
    if (enable_device_id < 0) {
        std::cerr << "GPU module isn't built for GPU" << std::endl;
        return -1;
    }
    cv::cuda::setDevice(enable_device_id);
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    std::cout << "GPU is ready, device ID is " << num_devices << "\n";


    cv::Mat a = cv::imread("00001.jpg");
    cv::Mat out;
//    TIC
    for (int i=0; i< 500 ; i++){
        out = cuda_hist_proc(a);
    }
//    TOC
//    cout<<"cuda time: "<<inf_time1<<endl;

    TIC
    for (int i=0; i< 500 ; i++){
        out = cpu_hist_proc(a);
    }
    TOC
    cout<<"cpu all time: "<<inf_time1<<endl;

    cv::imwrite("cpu.jpg", out);
    return 0;
}