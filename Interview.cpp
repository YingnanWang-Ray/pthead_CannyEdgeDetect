#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <string.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <pthread.h>
#include <cstdlib>
#include <ctime>
#include <windows.h>

#define pi 3.14159
#define NUM_THREADS 8

using namespace std;
clock_t start,Tend;
bool try_use_gpu = false;
int Test_count = 0;

struct thread_data {
    thread_data():img(), result(), guaSize(), hightThres(), lowThres(){}
    cv::Mat img;
    cv::Mat result;
    int guaSize;
    double hightThres;
    double lowThres;
};

void* ced(void* threadarg) {
    start = clock();
    struct thread_data* my_data;
    my_data = (struct thread_data*)threadarg;
    // 高斯滤波
    cv::Mat img = my_data->img;
    cv::Mat& result = my_data->result;
    int guaSize = my_data->guaSize;
    double hightThres = my_data->hightThres;
    double lowThres = my_data->lowThres;
    cv::Rect rect; // IOU区域
    cv::Mat filterImg = cv::Mat::zeros(img.rows, img.cols, CV_64FC1);
    img.convertTo(img, CV_64FC1);
    result = cv::Mat::zeros(img.rows, img.cols, CV_64FC1);
    int guassCenter = guaSize / 2; // 高斯核的中心 // (2* guassKernelSize +1) * (2*guassKernelSize+1)高斯核大小
    double sigma = 1;   // 方差大小
    cv::Mat guassKernel = cv::Mat::zeros(guaSize, guaSize, CV_64FC1);
    for (int i = 0; i < guaSize; i++) {
        for (int j = 0; j < guaSize; j++) {
            guassKernel.at<double>(i, j) = (1.0 / (2.0 * pi * sigma * sigma)) *
                (double)exp(-(((double)pow((i - (guassCenter + 1)), 2) + (double)pow((j - (guassCenter + 1)), 2)) / (2.0 * sigma * sigma)));
        }
    }
    cv::Scalar sumValueScalar = cv::sum(guassKernel);
    double sum = sumValueScalar.val[0];
    guassKernel = guassKernel / sum;
    for (int i = guassCenter; i < img.rows - guassCenter; i++) {
        for (int j = guassCenter; j < img.cols - guassCenter; j++) {
            rect.x = j - guassCenter;
            rect.y = i - guassCenter;
            rect.width = guaSize;
            rect.height = guaSize;
            filterImg.at<double>(i, j) = cv::sum(guassKernel.mul(img(rect))).val[0];
        }
    }
    cv::Mat guassResult;
    filterImg.convertTo(guassResult, CV_8UC1);

    // sobel算子计算梯度
    cv::Mat gradX = cv::Mat::zeros(img.rows, img.cols, CV_64FC1); // 水平梯度
    cv::Mat gradY = cv::Mat::zeros(img.rows, img.cols, CV_64FC1); // 垂直梯度
    cv::Mat grad = cv::Mat::zeros(img.rows, img.cols, CV_64FC1);  // 梯度幅值
    cv::Mat thead = cv::Mat::zeros(img.rows, img.cols, CV_64FC1); // 梯度角度
    cv::Mat locateGrad = cv::Mat::zeros(img.rows, img.cols, CV_64FC1); //区域
    // x方向的sobel算子
    cv::Mat Sx = (cv::Mat_<double>(3, 3) << -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
        );
    // y方向sobel算子
    cv::Mat Sy = (cv::Mat_<double>(3, 3) << 1, 2, 1,
        0, 0, 0,
        -1, -2, -1
        );
    // 计算梯度赋值和角度
    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1; j++) {
            // 卷积区域 3*3
            rect.x = j - 1;
            rect.y = i - 1;
            rect.width = 3;
            rect.height = 3;
            cv::Mat rectImg = cv::Mat::zeros(3, 3, CV_64FC1);
            filterImg(rect).copyTo(rectImg);
            // 梯度和角度
            gradX.at<double>(i, j) += cv::sum(rectImg.mul(Sx)).val[0];
            gradY.at<double>(i, j) += cv::sum(rectImg.mul(Sy)).val[0];
            grad.at<double>(i, j) = sqrt(pow(gradX.at<double>(i, j), 2) + pow(gradY.at<double>(i, j), 2));
            thead.at<double>(i, j) = atan(gradY.at<double>(i, j) / gradX.at<double>(i, j));
            // 设置四个区域
            if (0 <= thead.at<double>(i, j) <= (pi / 4.0)) {
                locateGrad.at<double>(i, j) = 0;
            }
            else if (pi / 4.0 < thead.at<double>(i, j) <= (pi / 2.0)) {
                locateGrad.at<double>(i, j) = 1;
            }
            else if (-pi / 2.0 <= thead.at<double>(i, j) <= (-pi / 4.0)) {
                locateGrad.at<double>(i, j) = 2;
            }
            else if (-pi / 4.0 < thead.at<double>(i, j) < 0) {
                locateGrad.at<double>(i, j) = 3;
            }
        }
    }

    // 梯度归一化
    double gradMax;
    cv::minMaxLoc(grad, &gradMax); // 求最大值
    if (gradMax != 0) {
        grad = grad / gradMax;
    }

    // 确定双阈值
    cv::Mat caculateValue = cv::Mat::zeros(img.rows, img.cols, CV_64FC1); // grad变成一维
    cv::resize(grad, caculateValue, cv::Size(1, (grad.rows * grad.cols)));
    // caculateValue.convertTo(caculateValue, CV_64FC1);
    cv::sort(caculateValue, caculateValue, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING); // 升序
    long long highIndex = img.rows * img.cols * hightThres;
    double highValue = caculateValue.at<double>(highIndex, 0); // 最大阈值

    double lowValue = highValue * lowThres; // 最小阈值
    // 3.非极大值抑制， 采用线性插值
    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1; j++) {
            // 八个方位
            double N = grad.at<double>(i - 1, j);
            double NE = grad.at<double>(i - 1, j + 1);
            double E = grad.at<double>(i, j + 1);
            double SE = grad.at<double>(i + 1, j + 1);
            double S = grad.at<double>(i + 1, j);
            double SW = grad.at<double>(i - 1, j - 1);
            double W = grad.at<double>(i, j - 1);
            double NW = grad.at<double>(i - 1, j - 1);
            // 区域判断，线性插值处理
            double tanThead; // tan角度
            double Gp1; // 两个方向的梯度强度
            double Gp2;
            // 求角度，绝对值
            tanThead = abs(tan(thead.at<double>(i, j)));
            switch ((int)locateGrad.at<double>(i, j)) {
            case 0:
                Gp1 = (1 - tanThead) * E + tanThead * NE;
                Gp2 = (1 - tanThead) * W + tanThead * SW;
                break;
            case 1:
                Gp1 = (1 - tanThead) * N + tanThead * NE;
                Gp2 = (1 - tanThead) * S + tanThead * SW;
                break;
            case 2:
                Gp1 = (1 - tanThead) * N + tanThead * NW;
                Gp2 = (1 - tanThead) * S + tanThead * SE;
                break;
            case 3:
                Gp1 = (1 - tanThead) * W + tanThead * NW;
                Gp2 = (1 - tanThead) * E + tanThead * SE;
                break;
            default:
                break;
            }
            // NMS -非极大值抑制和双阈值检测
            if (grad.at<double>(i, j) >= Gp1 && grad.at<double>(i, j) >= Gp2) {
                //双阈值检测
                if (grad.at<double>(i, j) >= highValue) {
                    grad.at<double>(i, j) = highValue;
                    result.at<double>(i, j) = 255;
                }
                else if (grad.at<double>(i, j) < lowValue) {
                    grad.at<double>(i, j) = 0;
                }
                else {
                    grad.at<double>(i, j) = lowValue;
                }

            }
            else {
                grad.at<double>(i, j) = 0;
            }
        }
    }

    // 4.抑制孤立低阈值点 3*3. 找到高阈值就255
    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1; j++) {
            if (grad.at<double>(i, j) == lowValue) {
                // 3*3区域找强梯度
                rect.x = j - 1;
                rect.y = i - 1;
                rect.width = 3;
                rect.height = 3;
                for (int i1 = 0; i1 < 3; i1++) {
                    for (int j1 = 0; j1 < 3; j1++) {
                        if (grad(rect).at<double>(i1, j1) == highValue) {
                            result.at<double>(i, j) = 255;
                            break;
                        }
                    }
                }
            }
        }
    }
    // 结果
    result.convertTo(result, CV_8UC1);
    Tend = clock();
    double period = (double)(Tend - start) / CLOCKS_PER_SEC;
    cout << "Complete in" << period * 1000 << "ms" << endl;
    //pthread_exit(NULL);
    return NULL;
}

//图像拼接处理Stitch
cv::Mat MergeImg(vector<cv::Mat> imgs) {
    cv::Mat result;
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create();
    cv::Stitcher::Status status = stitcher->stitch(imgs, result);	// 使用stitch函数进行拼接
    if (status != cv::Stitcher::OK)
    {
        cout << "Can't stitch images, error code = " << int(status) << endl;
    }
    cout << "Merge complete!" << endl;
    return result;
}

//多线程
cv::Mat Multi_thread(cv::Mat img) {
    //cv::imshow("img", img);
    //cv::waitKey(2);
    pthread_t tids[NUM_THREADS];
    struct thread_data td[NUM_THREADS];
    cv::Mat image[NUM_THREADS];
    cv::Mat result[NUM_THREADS];
    cv::Mat grayImage[NUM_THREADS];

    //image[0] = img.rowRange(0 * (img.rows / NUM_THREADS), 1 * (img.rows / NUM_THREADS) + 20);
    //image[0] = img.colRange(0 * (img.cols / NUM_THREADS), 1 * (img.cols / NUM_THREADS) + 2 * (img.cols / NUM_THREADS));
    image[0] = img.colRange(0 * (img.cols / NUM_THREADS), 1 * (img.cols / NUM_THREADS) + 2);
    cv::cvtColor(image[0], grayImage[0], CV_BGR2GRAY);

    for (int i = 1; i < NUM_THREADS - 1; i++)
    {
        //水平分割
        //img(cv::Rect(0, i * (img.cols / NUM_THREADS), img.rows, img.cols / NUM_THREADS)).copyTo(image[i]);
        //image[i] = img.rowRange(i * (img.rows / NUM_THREADS) - 20, (i + 1) * (img.rows / NUM_THREADS) + 20);
        
        //竖直分割
        //img(cv::Rect(i * (img.rows / NUM_THREADS) ,0, img.rows / NUM_THREADS, img.cols)).copyTo(image[i]);    //Method1
        //image[i] = img.colRange((i - 1) * (img.cols / NUM_THREADS), (i + 2) * (img.cols / NUM_THREADS));    //Method2
        image[i] = img.colRange(i * (img.cols / NUM_THREADS) - 2, (i + 1) * (img.cols / NUM_THREADS) + 2);
        cv::cvtColor(image[i], grayImage[i], CV_BGR2GRAY);
    }

    //image[NUM_THREADS - 1] = img.rowRange((NUM_THREADS - 1) * (img.rows / NUM_THREADS) - 20, NUM_THREADS * (img.rows / NUM_THREADS));
    //image[NUM_THREADS - 1] = img.colRange((NUM_THREADS - 1) * (img.cols / NUM_THREADS) - 2 * (img.cols / NUM_THREADS), NUM_THREADS * (img.cols / NUM_THREADS));
    image[NUM_THREADS - 1] = img.colRange((NUM_THREADS - 1) * (img.cols / NUM_THREADS) - 2,img.cols);
    cv::cvtColor(image[NUM_THREADS - 1], grayImage[NUM_THREADS - 1], CV_BGR2GRAY);

    for (int i = 0; i < NUM_THREADS; ++i)
    {
        td[i].img = grayImage[i];
        td[i].result = result[i];
        td[i].guaSize = 3;
        td[i].hightThres = 0.8;
        td[i].lowThres = 0.5;
        int ret = pthread_create(&tids[i], NULL, &ced, (void*)&td[i]); //参数：创建的线程id，线程参数，线程运行函数的起始地址，运行函数的参数  
        if (ret != 0) //创建线程成功返回0  
        {
            cout << "pthread_create error:error_code=" << ret << endl;
        }
    }

    /*等待全部子线程处理完毕*/
    for (size_t i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(tids[i], NULL);
    }

    /*图像拼接*/
    //去除边界
    vector<cv::Mat> imgs;
    imgs.resize(NUM_THREADS);
    imgs[0] = td[0].result.colRange(0, td[0].result.cols - 2);
    for (int i = 1; i < NUM_THREADS - 1; i++) {
        imgs[i] = td[i].result.colRange(2, td[i].result.cols - 2);
    }
    imgs[NUM_THREADS - 1] = td[NUM_THREADS - 1].result.colRange(2, td[NUM_THREADS - 1].result.cols);

    cv::Mat O_i = imgs[0];
    for (int i = 1; i < NUM_THREADS; ++i) {
        cv::hconcat(O_i, imgs[i], O_i);//vconcat(hconcat)
    }
    //debug输出图像测试
    /*
    string outname = "D:\\Code\\mihoyo\\Interview\\temp\\";
    outname += (char)(Test_count + 48);
    outname += ".png";
    Test_count++;
    cv::imwrite(outname, O_i);
    */
    cv::imshow("Out_image", O_i);
    cv::waitKey(3000);
    return O_i;

    //竖直分割拼接
    /*
    cout << "Output image" << endl;
    cv::Mat Out_image = td[0].result;
    for (int i = 1; i < NUM_THREADS; ++i) {
        cv::hconcat(Out_image, td[i].result, Out_image);//vconcat(hconcat)
    }
    cv::imshow("Out_image", Out_image);
    cv::waitKey(3000);
    return Out_image;
    */
    //预留接口——后期图片拼接边缘处理
    /*
    vector<cv::Mat> imgs;
    cv::Mat Out_image; 
    for (int i = 1; i < NUM_THREADS - 1; i++) {
        imgs.push_back(td[i].result);
    }
    Out_image = MergeImg(imgs);
    cv::imshow("Result", Out_image);
    return Out_image;
    */
}


int main() {
    //debug单文件测试
    //cv::Mat img = cv::imread("D:\\Code\\mihoyo\\Interview\\PNG\\2.png");
    //Multi_thread(img);

    //debug图像拼接
    //彩色图片简单拼接测试
    /*
    cv::Mat img0 = cv::imread("D:\\Code\\mihoyo\\Interview\\PNG\\campus_000.jpg");
    cv::Mat img1 = cv::imread("D:\\Code\\mihoyo\\Interview\\PNG\\campus_001.jpg");
    vector<cv::Mat> im;
    im.push_back(img0);
    im.push_back(img1);
    cv::imshow("Test", MergeImg(im));
    cv::waitKey(0);
    */

    //多图片测试
    /*
    cv::Mat img0 = cv::imread("D:\\Code\\mihoyo\\Interview\\temp\\1.png");
    cv::Mat img1 = cv::imread("D:\\Code\\mihoyo\\Interview\\temp\\2.png");
    cv::Mat img2 = cv::imread("D:\\Code\\mihoyo\\Interview\\temp\\3.png");
    cv::Mat img3 = cv::imread("D:\\Code\\mihoyo\\Interview\\temp\\4.png");
    cv::Mat img4 = cv::imread("D:\\Code\\mihoyo\\Interview\\temp\\5.png");
    cv::Mat img5 = cv::imread("D:\\Code\\mihoyo\\Interview\\temp\\6.png");
    vector<cv::Mat> im;
    im.push_back(img0);
    im.push_back(img1);
    im.push_back(img2);
    im.push_back(img3);
    im.push_back(img4);
    im.push_back(img5);
    cv::Mat tt = MergeImg(im);
    cv::imshow("123", tt);
    cv::waitKey(0);
    */


    /*指定文件目录*/
    //debug
    //cv::String pngfolder = "D:\\Code\\mihoyo\\Interview\\PNG";
    //cv::String outfolder = "D:\\Code\\mihoyo\\Interview\\Result";
    //输入目标文件夹名称

    cout << "Please input the folder path:" <<"          " << "Example: D:\\Code\\mihoyo\\Interview\\PNG" << endl;
    cv::String pngfolder;
    cin >> pngfolder;
    cout << "Done" << endl << endl ;

    cout << "Please input the output folder path:" << "          " << "Example: D:\\Code\\mihoyo\\Interview\\Result" << endl;
    cv::String outfolder;
    cin >> outfolder;
    cout << "Done" << endl << endl;

    Sleep(1000);
    system("cls");

    pngfolder += "\\*.png";
    outfolder += "\\";

    //文件夹读取文件
    //vector<cv::Mat> pngs;     // 数组储存文件名
    vector<cv::String> fn;
    cv::glob(pngfolder, fn, false);
    //for (auto x : fn) { cout << x << endl; }      //debug 输出文件名
    size_t count = fn.size();
    for (int i = 0; i < count; i++) {
        //pngs.push_back(cv::imread(fn[i]));//存入pngs数组
        //cv::imshow("pic", pngs[i]);//show图片
        cout << "Begin detecting image:" << fn[i] << endl;
        cv::Mat img = cv::imread(fn[i]);
        size_t position = fn[i].find_last_of('\\') + 1;
        string outname = outfolder + fn[i].substr(position);
        cv::imwrite(outname, Multi_thread(img));
        cout << "Complete: " << outname << endl;
        cout << endl << endl;
    }
    system("Pause");
    return 0;
}