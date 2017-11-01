---
layout: post
title: 【Learning Opencv3】之背景差分
date: 2017-09-22 12:00:00
category: Opencv
tags:
- Opencv3
- 图像处理
---
# Overview of Background Subtraction

在很多场景下，相机位置是固定的，所以可以用背景差分法来作为一种处理手段。再加上自身方法的简单性，背景差分法可以算是一种重要的图像处理操作，有很多实际的应用，尤其是在视频监控中。Toyama,Krumm,Brumitt和Meyers对很多背景差分的方法做了一个很好的综述和比较[Toyama99]。为了实现背景差分，我们首先必须对背景进学习。
<!--more-->

背景学习完成后，将当前图像与背景模型进行比较，已知的背景部分将被减去。减去背景后剩下的东西就是新的前景目标。

当然，“背景”是一个很难定义的概念，要具体问题具体分析。比如，你所关注的是一条高速公路，或许平均的车流应该被视为背景。通常，背景被认为是在感兴趣的时期内保持静态或周期性的场景的任何静态或周期性移动的部分。整体效果可能会有时变的部分，例如在早上和晚上由于风吹，树叶会摇动，而在中午则保持静止。可能遇到的两个常见但实质上不同的环境类别是室内和室外场景。我们对在这两种场景下都能发挥作用的工具感兴趣。

在这一章中，我们将首先讨论典型背景模型的缺点，然后会把重点放在讨论高级场景模型上。在这种情况下，我们提出一种快速的方法，它主要适用于照明变化不大的室内静态背景场景。然后我们介绍一下“codebook”方法，这种方法稍微慢一点，但是可以适应室内室外场景。它允许背景有一些周期性运动（比如树在风中的摆动），也允许光照缓慢或者周期性变化。这种方法在学习背景的过程中也能容忍偶尔的前景对象移动。我们将在清楚前景目标检测的内容中重点讨论连通区域。然后，我们会对快速背景方法和codebook背景方法做一个比较。这章主要讨论OpenCV库可用的用于背景差分的两种现代算法
。这些算法使用这章所涉及到的一些原则，当然也包括一些额外的扩展和补充，以便使他们更好地适应于实际应用。

# Weaknesses of Background Subtraction

尽管这里提到的背景建模方法在简单场景中效果确实很好，但是这其中有一个假设经常得不到满足，从而使检测效果恶化，即：图像中的所有像素之间是统计独立的。值得注意的是，我们所描述的这种方法只是学习单个像素所经历的变化，从而建立一个模型。而没有考虑任何其它相邻像素。为了将邻域像素考虑进去，我们可以学习多个模型。比如可以把基本的独立像素模型扩展到包含对邻域像素亮度的基本感觉（rudimentary sense）。这种情况下，如果邻域像素的值分别是亮或者是暗，我们就用邻域像素的亮度来区分。然后，我们将对一个像素点有效地学习两个模型：一个模型对应于其邻域像素点的值是亮的情况，另一个模型对应于其邻域像素点的值是暗的情况。这样做的话，我们的模型就会把像素周围的环境也考虑进去。因为根据邻域像素值是亮还是暗，基本像素需要有两个不同的值来对应，这也意味着两倍存储空间和更大的计算量问题。我们也要两倍的数据来填充这个两种状态的模型。我们可以把“亮”背景和“暗”背景的概念推广到单一像素值强度和邻域像素值强度的多维直方图，或者可以通过在很短的时间内来完成这些所有的事情来使它更复杂。当然，这个在时间上和空间上都很臃肿的模型需要更多的内存，更多收集到的数据样本，以及更多的计算资源。

因为这些额外的开销，一般不太用更加复杂的模型。我们应该把资源更加有效地投入到清除假阳性像素的任务中，这些假阳性像素是违反了独立像素假设时产生的。这种清理通常采取图像处理中用来消除杂散斑块像素的操作(`cv::erode()`, `cv::dilate()`, `cv::floodFill()`)。我们在这章还会再次使用连通区域的方法，但是，现在我们要先假定像素之间是独立变化的。


# Scene Modeling

我们如何定义背景和前景？如果我们正在关注一个停车场，一辆车开了进来，这辆车就是一个新的前景目标。但是要把这辆车一直当作前景吗？对于一袋可以移动的垃圾呢？它在两个地方将被显示为前景：垃圾被移动到的地方和它被移走留下的洞。如果我们正在为一个黑屋子建模，突然有人打开了灯，整个房子应该变成前景吗？为了回答这些问题，我们需要更高级别的场景模型。我们在前景状态和背景状态之间定义多个级别，以及基于时间方法：将不动的前景块归并到背景中。当场景有了全局的变化时，我们也必须检测到并且创造一个新的模型。

总之，一个场景模型可能包含很多层，从“新前景”到较旧的前景再到背景。 可能还需要一些运动检测。当一个目标移动的时候，我们可以辨识出它的“正”的部分（目标的新位置）和“负”的部分（目标旧的位置，即我们所说的那个“洞”）。

这样，一个新的前景目标将被放进“新前景”目标级，并且被标记为正目标或者是一个“洞”。在没有前景目标的区域我们可以继续更新我们的北京模版。如果一个前景目标在给定的时间内没有移动，将被降级为“较旧的前景”，它的像素统计信息将被临时学习直到它学到的模型加入到背景模型中。

对于全局变化检测，例如打开房间的灯，我们可以用全局帧间差分法。举个例子，如果在一瞬间很多像素发生了变化，我们可以将它分类为全局变化而不是局部变化，然后转而在新的情况下使用不同的模型。

## A Slice of pixels

在我们着手建立像素变化模型之前，先来看看随着时间的推移一张图片中的像素是如何变化的。考虑一个朝向窗外的相机，窗外是一棵树，在风中摆动。图15-1显示了一行像素在60帧图像中的像素值变化情况。我们希望对这些波动进行建模。在建模之前，我们需要再多嘴来讨论一下如何来对这一行像素进行取样，因为这对于特征跟踪和调试都很有帮助。
![Fig.15-1](assets/post_images/Fluctuation_of_a_line_of_pixels.jpg)

因为在很多情况下都会出现这种需求，借助OpenCV抽取任意一行像素是非常简单的。完成这个任务的对象叫做行迭代器。行迭代器，`cv::LineIterator`，是一个对象，一旦被实例化，就可以告诉我们关于图像中一行的所有点的信息。

我们需要做的第一件事就是实例化一个行迭代器对象，通过`cv::LineIterator`结构体来干这件事。

```c++
cv::LineIterator::LineIterator(
  const cv::Mat& image,            //需要被迭代的图像
  cv::Point      pt1,              //迭代器的初始点
  cv::Point      pt2,              //迭代器的结束点
  int            connectivity=8,   //连通性，4连通或者8连通
  int            left_to_right=0   //迭代方向，1表示相反的方向
);
```

这里，输入图像可以是任意类型，任意通道数量。`pt1`和`pt2`是直线的端点。连通性可以是4连通（直线的像素可以在上下左右四个方向移动）或者8连通（在4连通的基础上再加上沿着对角线的四个方向）。最后，如果`left_to_right`被设置成0，`line_iterator`将会从`pt1`到`pt2`进行浏览；反之，它将从最左端到最右端进行浏览。

然后可以将迭代器递增，指向给定端点之间的线上的每个像素。我们通常通过`cv::LineIterator::operator++()`来增加迭代器。所有的通道同时有效。假设我们的行迭代器叫做`line_iterator`，我们可以通过解引用迭代器（e.g.`*line_iterator`）来访问当前像素点。在这里需要注意：`cv::LineIterator::operator*()`的返回类型不是指向内置OpenCV向量类型的指针，而是一个`uchar*` 指针。这意味着你通常想要把自己的值转换成类似`cv::Vec3f*`的值（对于矩阵图像来说自然无所谓）。

有这么方便的一个工具在手头，我们可以从一个文件中提取一些数据。Example 15-1的程序从一个视频文件中生成Figure 15-1中所示的数据。

```c++
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace std;

void help( argv ) {
  cout<<"\n"
  << "Read out RGB pixel values and store them to disk\nCall:\n"
  << argv[0]<<" avi_file\n"
  << "\n This will store to files blines.csv, glines.csv and rlines.csv\n\n"<<endl;
}

int main( int argc, char** argv ) {

  if(argc != 2) { help(); return -1; }
  cv::nameWindow( argv[0], CV_WINDOW_AUTOSIZE );

  cv::VideoCapture cap;
  if((argc < 2)|| !cap.open(argv[1]))
  {
    cerr<<"Couldn't open video file" << endl;
    help();
    cap.open(0);
    return -1;
  }

  cv::Point pt1(10,10), pt2(30,30);
  int max_buffer;
  cv::Mat rawImage;
  ofstream b,g,r;
  b.open("blines.csv");
  g.open("glines.csv");
  r.open("rlines.csv");

  // MAIN PROCESSING LOOP：
  //
  for (; ;) {
    cap >> rawImage;
    if( !rawImage.data ) break;

    cv::LineIterator it( rawImage, pt1, pt2, 8);
    for ( int j = 0; j < it.count; ++j,++it) {
      b << (int)(\*it)[0] << ",";
      g << (int)(\*it)[1] << ",";
      r << (int)(\*it)[2] << ",";
      (\*it)[2]=255;          // 将这个样本标记为红色
    }

    cv::imshow( argv[0], rawImage );
    int c = cv::waitKey(10);
    b << "\n"; g <<"\n"; r << "\n";
  }

  // CLEAN UP:
  //
  b << endl; g << endl; r << endl;
  b.close(); g.close(); r.close();
  cout << "\n"
  << "Data stored to files: blines.csv, glines.csv and rlines.csv\n\n"
  << endl;
}
```
在 Example 15-1中，我们一个像素一个像素地移动，一次移动一个像素，一次处理一个像素。另一种常见做法是创建一个缓冲区，将整行拷贝到缓冲区，再去处理缓冲区的数据。这种情况下，缓冲区副本看起来像以下内容：

```c++
cv::LineIterator it( rawImage, pt1, pt2, 8 );

vector<cv::Vec3b> buf( it.count);

for (int i = 0; i < it.count; i++,++it) {
  buf[i] = &( (const cv::Vec3b*) it );
}
```

这种方法的主要优势：如果图像 rawImage 不是一个无符号字符类型，该方法以稍微更清洁的方式将数据转换为适当的向量类型。

现在，我们准备采取一些方法来建模像素波动的种类。当我们从简单到复杂的模型转变时，我们将把注意力集中在那些实时的、合理的内存限制下运行的模型上。

## Frame Differencing
最简单的背景差分方法就是从一帧图像（可能是后面几帧）里减去另一帧图像，然后标出足够大的差值作为前景。这个过程可以抓取移动目标的边缘。为简单起见，假设我们有三个单通道图像：`frameTime1`,`frameTime2`和`frameForeground`。`frameTime1`是过去的灰度图像，`frameTime2`是当前的灰度图像。我们可以根据下面的代码来检测在`frameForeground`中前景差的绝对值：

```c++
cv::absdiff(
  frameTime1,            // 第一个输入矩阵
  frameTime2,            // 第二个输入矩阵
  frameForeground        // 结果矩阵
);

```
由于像素值经常有一些噪声和波动，我们应该忽略一些小的差值（比如说忽略小于15的差值），标记出那些比较大的差值：

```c++
cv::threshold(
  frameForeground,          // 输入图像
  frameForeground,          // 输出图像
  15,                       // 阈值
  255,                      // 前向操作最大值
  cv::THRESH_BINARY         // 要用的阈值类型
);

```

`frameForeground`将候选前景目标标记为255，将背景像素标记为0。如前面所述，我们需要清除一些小的噪声区域；可以借助`cv::erode()`来完成这个任务，也可以用联通区域的方法。对于彩色图像，我们可以分别对其每一个颜色通道执行上面相同的代码，然后用`cv::max()`函数来合并这些通道。对于大多数不仅仅满足于指示运动区域的应用而言，这未免有些太简单了。对于更多有效的背景模型，我们需要关注一些统计信息，比如期望和场景中像素的平均差。


# Averaging Background Method

平均背景方法主要学习每个像素的均值和标准差作为它的背景模型。

考虑之前我们讨论的抽取一行像素的情况。我们可以根据平均和平均差异来表示整个视频中每个像素的变化，而不是绘制每一帧的一系列值。在相同的视频中，前景目标（一只手）出现在相机前面。这个前景目标的亮度值与背景中的天空和树的亮度值差异比较大。手的亮度值也表示在下图中

![Fig.15-2](assets/post_images/Averaging_Method_with_High_Low_Thresholds.jpg)

平均背景方法主要利用四个OpenCV 程序：
+ `cv::Mat::operator+=()` 随着时间的推移累加图像；
+ `cv::absdiff()` 随着时间的推移累加帧间差值；
+ `cv::inRange()` 分割图像（一旦背景模型学习完成），分割成前景图像和背景图像；
+ `cv::max()` 将不同颜色通道的分割区域融合成单一通道的图像。
由于这个代码例子比较长，我们将分段描述。

首先，我们为所需的各种scratch和统计图像创建指针。根据它们所指向的图像的类型对这些指针进行分类是很有必要的。

*Example 15-2. Learning a background model to identify foreground pixels*

```c++
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace std;

// Global storage
//
// Float, 3-channel images
//
cv::Mat IavgF, IdiffF, IprevF, IhiF, IlowF;
cv::Mat tmp, tmp2;

// Float, 1-channel images
//
vector<cv::Mat> Igray(3);      //Igray包含了3个重复地执行了值初始化的对象，即有3个Mat类型的对象
vector<cv::Mat> Ilow(3);
vector<cv::Mat> Ihi(3);

// Byte, 1-channel images
//
cv::Mat Imaskt;

// Counts number of images learned for averaging later
//
float Icount;

```
接下来，我们创建一个单独的调用来分配所有必要的中间图像。方便起见，我们从视频中拿出一个单独的图像，用它来传递中间图像的尺寸：

```c++
// I只是一个样本图像
// (passed in for sizing)
//
void AllocateImages( const cv::Mat& I) {

  cv::Size sz = I.size();

  IavgF  = cv::Mat::zeros( sz, CV_32FC3 );  // CV_32FC3 表示32位float类型，3通道
  IdiffF = cv::Mat::zeros( sz, CV_32FC3 );
  IprevF = cv::Mat::zeros( sz, CV_32FC3 );
  IhiF   = cv::Mat::zeros( sz, CV_32FC3 );
  IlowF  = cv::Mat::zeros( sz, CV_32FC3 );
  Icount = 0.00001; // 防止分母为0；

  tmp    = cv::Mat::zeros( sz, CV_32FC3 );
  tmp2   = cv::Mat::zeros( sz, CV_32FC3 );
  Imaskt = cv::Mat( sz, CV_32FC1);

}

```
在接下来的一部分代码中，我们学习累加的背景图像以及帧间图像差（用于学习图像像素标准差的一种计算上更快速的代理（近似））绝对值的累加和。这种方法通常需要30到1000帧图像，有时在每一秒只取几帧图像，有时取所有可用的帧。这个程序被3彩色通道8bit的图像调用。

```c++
// Learn the background statistics for one more frame
// I is a color sample of the background, 3-channel, 8u
//
void accumulateBackground( cv::Mat& I ) {

  static int first = 1;         //
  I.converTo( tmp, CV_32F );    // 转换成float类型
  if (!first) {
    iavgF += tmp;
    cv::absdiff( tmp, IprevF, tmp2 );
    IdiffF += tmp2;
    Icount += 1.0;
  }
  first = 0;
  IprevF = tmp;
}

```
我们首先用`cv::Mat::convertTo()`将原始的背景8-bit-per-channel,3颜色通道的图像转换成浮点，三通道图像。然后将原始的浮点图像累加到`IavgF`中。接着，我们用`cv::absdiff()`计算帧间差的绝对值并将其累加到图像`IdiffF`中。每一次我们累加图像，都会增加图像计数`Icount`，`Icount`是一个全局变量，后面用来求平均数。

一旦我们累加了足够的图像帧，就将其变换成背景的统计模型；也就是说，计算每个像素的期望和标准差：

```c++
void createModelsfromStats() {
  IavgF  * =(1.0/Icount);
  IdiffF * =(1.0/Icount);

  // Make sure diff is always something
  //
  IdiffF += cv::Scalar( 1.0, 1.0, 1.0 );
  setHighThreshold( 7.0 );
  setLowThreshold( 6.0 );
}

```
在这部分，通过除以累加图像的总数，我们用`cv::Mat::operator*=()`来计算平均背景和图像差的绝对值。预防起见，我们要保证平均差值的图像数量至少是1；我们需要在计算前景背景阈值时缩放这个因子，并希望避免这两个阈值变得相等的退化情况。

接下来的第二个程序，`setHighThreshold()` 和`setLowThreshold()`是*utility functions*，它们根据帧间平均绝对差（FFAAD）来设置一个阈值。FFAAD是我们判断所观察到的变化是否显著的一个基本指标。例如，调用`setHighThreshold(7.0)`，修正一个阈值，使得对于该像素的平均值高于 FFAAD 的7倍的任何值被认为是前景；同理，`setLowThreshold(6.0)`设置一个阈值范围，该范围使得对于该像素的平均值低于FFAAD的6倍。围绕当前像素点在这个范围内，目标被认为是背景。这些阈值函数如下：
```c++
void setHighThreshold( float scale) {
  IhiF = IavgF + (IdiffF * scale);
  cv::split( IhiF, Ihi );
}
void setLowThreshold( float scale ) {
  IlowF = IavgF - (IdiffF * scale);
  cv::split( IlowF, Ilow );
}
```

在`setLowThreshold()`和`setHighThreshold()`中，在`IavgF`上加减这些范围之前，我们首先要缩放差值图像（the FFFAAD）。这个步骤中，我们通过`cv::split()`对图像中的每个通道设置`IhiF`
和`IlowF`范围。

一旦我们有了背景模型，连同高阈值和低阈值，我们用这些来把图像分割为前景和背景。我们通过调用以下代码来执行分割任务：

```c++
// Create a binary: 0,255 mask; 255表示前景像素
// I 输入图像，3通道，8u
// Imask Mask image to be created, 1通道 8u
//
void backgroundDiff(cv::Mat& I, cv::Mat& Imask) {
  I.convertTo( tmp, CV::F32 );    // To float
  cv::split( tmp, Igray );

  // Channel 1
  //
  cv::inRange( Igray[0], Ilow[0], Ihi[0], Imask );

  // Channel 2
  //
  cv::inRange( Igray[1],Ilow[1], Ihi[1], Imaskt );
  Imask = cv::min( Imask, Imaskt );

  //Channel 3
  //
  cv::inRange( Igray[2], Ilow[2], Ihi[2], Imaskt );
  Imask = cv::min( Imask, Imask );

  // Finally, invert the results
  //
  Imask = 255-Imask;
}

```
这个函数首先通过调用`cv::Mat::convertTo()`将输入图像`I`(要被分割的图像)转换成浮点型图像。然后用`cv::split()`将3通道图像转换成独立单通道图像。接着，我们要通过`cv::inRange()`函数检查这些颜色通道平面，看它们是否在平均背景像素的高低范围之内。当它在背景图像高低范围之内时，设置相应8-bit深度图相应像素值为最大(255)，反之，则将像素值设置为0。对于每个颜色通道，我们把分割结果 **逻辑与** 到一个mask图像`Imask`中，因为在每一个颜色通道中像素值强烈的变化都被认为是判断其是前景像素的证据。最后，我们用`cv::operator-()`来反转`Imask`，因为前景应该是超出范围之外的值，而不是在范围之内的值。mask图像就是输出结果。

将它们都放在一起，我们可以定义main()函数，来读取视频并且建立背景模型。例如，首先我们在训练模式中跑视频，直到用户敲了空格键，之后程序工作在检测前景模式中，检测出的前景目标用红色高亮标出：

```c++
void help(argv) {
  cout << "\n"
  << "Train a background model on incomimng video, then run the model\n"
  << argv[0]<<" avi_file\n"
  << endl;
}

int main( int argc, char** argv ){

  if (argc != 2) {
    help( argv ); return -1;
  }

  cv::nameWindow( argv[0], cv::WINDOW_AUTOSIZE );
  cv::VideoCapture cap;

  if ((argc < 2)|| !cap.open(argv[1])) {
    cerr << "Couldn't open video file" << endl;
    help();
    cap.open(0);
    return -1;
  }

  // FIRST PROCESSING LOOP (TRAINING):
  //
  while(1){
    cap >> image;
    if (!image.data) {
      exit(0);
    }

    accumulateBackground( image );

    cv::imshow( argv[0],rawImage );
    if( cv::waitKey(7) == 0x20 ) break;  
  }

  // We have all of our data, so create the models
  //
  createModelsfromStats();

  // SECOND PROCESSING LOOP (TESTING):
  //
  cv::Mat mask;
  while (1) {
    cap >> image;
    if( !image.data ) exit(0);

    backgroundDiff( image, mask );

    // A simple visualization is to write to the red Channel
    //
    cv::split( image, Igray );
    Igray[2] = cv::max( mask, Igray[2] );
    cv::merge( Igray, image );
    if( cv::waitKey(7) == 0x20 ) break;
  }
  exit(0);
}

```

我们已经见识过了学习背景场景、分割前景目标的一种简单方法。这个场景中不能有移动的背景元素，否则该方法不会见效（场景中若有摆动的窗帘和摇动的树木，或者是其它一些能够产生双峰或者多峰特征，这种方法会失效）

## Accumulating Means, Variances, and Covariances

平均背景方法仅仅描述了利用累加操作 `cv::Mat::operator+=()` 如何去做本质的事情：把一簇数据加起来然后归一化成平均值。均值是一个很方便的统计量，但是它一个经常被忽略的优势是它可以进行递增运算。这就意味着我们无需在分析之前就把所有数据全部加起来，我们可以边递增边计算。我们现在考虑一个稍微更复杂的模型，也可以通过这种方式在线计算。

我们的下一个模型将通过计算该变化的高斯模型来表示像素内的强度（ 或颜色 ）变化。一个一维高斯模型由一个均值和一个方差来表征。而对于d维模型，均值将是一个d维向量，以及一个d^2个元素的矩阵，该矩阵不仅代表着d维的方差，还表示协方差。协方差表示每个维度之间的相关性。

每一个统计量————均值、方差和协方差都可以通过增量计算得到。给定一股输入图片流，我们可以定义三个函数来累加需要的数据，这三个函数实际上可以将那些累加和转换成模型参数。

下面的代码假定存在几个全局变量：

```c++
cv::Mat sum;
cv::Mat sqsum;
int image_count = 0;
```

### Computing the mean with cv::Mat::operator+=()

正如我们在之前的例子所看到的那样，计算像素均值的最好方法就是用`cv::Mat::operator+=()`把所有的加起来，然后再除以图像的总数来获得均值：

```c++
void accumulateMean(cv::Mat& I) {
  if (sum.empty) {
    sum = cv::Mat::zeros(I.size(), CV_32FC(I.channels()));
  }
  I.convertTo( scratch, sum.type() );
  sum += scratch;
  image_count++;
}
```

之前的函数，`accumulateMean()`，将会被每个输入图像所调用。一旦所有将要被用于背景建模的图像被计算完毕，你就可以调用下一个函数了，即`computeMean()`。结果是可以得到一幅包含整个输入图片集的每个像素均值的单一图像。

```c++
cv::Mat& computeMean(cv::Mat& mean){
  mean = sum / image_count;
}
```

### Computing the mean with cv::accumulate()

OpenCV 提供其它函数，基本上和使用`cv::Mat::operator+=()`类似，但有两个重要区别。第一，它会自动执行`cv::Mat::convertTo()`的功能（从而消除了对scratch image的需求）；第二，它允许使用图像mask。这个函数是`cv::accumulate()`。当计算一个背景模型时，使用图像mask是非常有用的，因为你经常有一些其他信息来判断图像的某些部分不应该包含在背景模型中。例如，你可能正在为一个高速公路背景建模，或者其它一些均匀着色区域，我们可以根据颜色立即判定某些目标不属于背景。~~这种对物体的分类，在现实世界中很有帮助，在现实世界中，在完全没有前景对象的情况下很少或没有机会进入场景。~~（烂翻译）This sort of thing can be very helpful in a real-world situation in which there is little or no opportunity to get access to the scene in the complete absence of Foreground objects.

累加函数的原型：
```c++
void accumulate(
  cv::InputArray        src,                 // Input, 1 or 3 channels, U8 or F32
  cv::InputOutputArray  dst,                 // 输出图像，F32 or F64
  cv::InputArray        mask = cv::noArray() // Use src pixel if mask pixel != 0
);
```
`dst`表示已累加的结果，`src`是将要被累加的图像。`cv::accumulate()` admits an optional mask. If present, only the pixels in dst that correspond to nonzero elements in mask will be updated.

有了`cv::accumulate()`，前面的`accumulateMean()`函数可以被简化成:

```c++
void accumulateMean(cv::Mat& I) {
  if ( sum.empty ) {
    sum = cv::Mat::zeros( I.size(), CV_32FC(I.channels()));
  }
  cv::accumulate( I, sum);
  image_count++;
}

```
### Variation:Computing the mean with cv::accumulateWeighted()

另一种有用的选择是 *running average*， 它的定义是：

$$ acc(x,y) = (1-\alpha)\cdot acc(x,y)+\alpha \cdot image(x,y) $$

对于常数 $\alpha$，*running average* 不等于用`cv::accumulate()`或者`cv::Mat::operator+=()`累加的结果。为了说明这点，只需要考虑下面一种情况：将3个数（2，3和4）进行相加，$\alpha$ 设为0.5。如果我们用`cv::accumulate()`来累加，和是9，平均值是3。如果用`cv::accumulateWeighted`，第一个和为 $0.5 \cdot 2 + 0.5 \cdot 3 = 2.5 $，再加上第三个变量，结果是$0.5 \cdot 2.5 + 0.5 \cdot 4 = 3.25$。第二种情况的均值更大的原因是越靠近当前的值其贡献越大，权重因子更大。这种*running average* 也叫做*tracker*。你可以考虑参数 $\alpha$ 设置前一帧的影响所需的时间尺度 - 越小，过去帧的影响消失得越快。

为了累加所有图像的*running averages*，我们用OpenCV函数`cv::accumulateWeighted()`:
```c++
void accumulateWeighted(
  cv::InputArray         src,                  // Input, 1 or 3 channels, U8 or F32
  cv::InputOutputArray   dst,                  // Result image, F32 or F64
  double                 alpha,                // Weight factor applied to src
  cv::InputArray         mask = cv::noArray()  // Use src pixel if mask pixel != 0
);
```
`dst`表示已累加的结果，`src`是将要被累加的图像。`cv::accumulate()` admits an optional mask. If present, only the pixels in dst that correspond to nonzero elements in mask will be updated.

### Finding the variance with the help of cv::accumulateSquare()
我们也可以累加平方图像，如此这般，我们就可以快速地计算单个像素的方差。方差的定义：

$$ \sigma^2=\frac{1}{N} \sum_{i=0}^{N-1}(x_i-\overline x)^2 $$

其中 $\overline x$ 是 $x$ 的 $N$ 个样本的均值。这个公式有一个问题，即它需要传输两边图像，第一遍计算 $\overline x$ ， 第二遍计算 $\sigma^2$ 。将上面公式变形可得到：

$$ \sigma^2=(\frac 1 N \sum_{i=0}^{N-1}x_i^2)-(\frac 1 N \sum_{i=0}^{N-1}x_i)^2 $$

用这个形式，我们只需要传递一次图像既可以计算每个像素的值也可以计算它们的平方。然后，每个像素的方差就是平方的均值减去均值的平方。基于这个思想，我们可以定义一个累加函数和一个计算均值的函数。和均值一样，第一件要做的事就是对输入图像逐个元素的求平方，然后再用其它函数（比如`sqsum += I.mul(I)`）去计算方差。然而，这种方法也有缺点，最显著的一个缺点是`I.mul(I)`不能进行任何内置类型的转换（正如我们在`cv::Mat::operator+=()`操作中也做不到的那样）。因此，一个8-bit的数组元素，一被平方，将不可避免的导致泄漏。然而，和`cv::accumulate()`一样，OpenCV 给我们提供了一个函数，可以在一个方便的包里做到所有我们所需要的事情，即`cv::accumulateSquare()`:
```c++
void accumulateSquare(
  cv::InputArray              src,                 // 输入图像，1 或 3 通道，U8 或者 F32
  cv::InputOutputArray        dst,                 // 输出图像，F32 或者 F64
  cv::InputArray              mask = cv::noArray() // Use src pixel if mask pixel != 0
);
```
有了`cv::accumulateSquare()`的帮助，我们可以写一个函数来计算我们要得到方差所需要的信息：

```c++
void accumulateVariance(cv::Mat& I) {
  if (sum.empty) {
    sum = cv::Mat::zeros( I.size(), CV_32FC(I.channels()));
    sqsum = cv::Mat::zeros(I.size(), CV_32FC(I.channels()));
  }
  cv::accumulate( I, sum );
  cv::accumulateSquare( I, sqsum );
  image_count++;
}
```
相应的计算函数如下：

```c++
// note that 'variance' is sigma^2
//
void computeVariance(cv::Mat& variance)
{
  double one_by_N = 1.0 / image_count;
  variance        = one_by_N * sqsum - (one_by_N * one_by_N) * sum.mul(sum);
}
```
### Finding the covariance with cv::accumulateWeighted()

在多通道图像中每个通道的方差都携带着重要的信息（我们所希望未来的背景像素和实际观察到的平均值之间的相似度）。然而，对于背景和我们的“期望”来说，这仍然是一个非常简单的模式。这里要额外介绍一个很重要的概念——协方差。协方差表示各个通道方差之间的相互关系。

例如，我们的背景可能是一个海洋的场景，自然，我们希望在红色通道里的方差很小，在绿色和蓝色通道里的方差稍微大一些。我们的直觉是海的颜色只有一种，我们所看到的方差是由光照强度变化引起的，因此我们可能认为：绿色通道的强度变化和蓝色通道中的颜色变化应该是保持一致的。这个直觉所引发的推论是：如果蓝色通道强度的增加没有随着绿色通道强度的增加而增加，我们就不能把这部分认为是背景了。这个直觉可以用协方差的概念来说明。

在 *Figure 15-3*，我们可以看到海洋背景中某个像素的绿色通道和蓝色通道的亮度值分布情况。左边只计算了方差，右边，两通道的协方差也被计算了。可见右边的模型更加贴合数据分布。
![Figure 15-3](assets/post_images/Covariance.jpg)

协方差按下列公式计算

$$ Cov(x,y)=(\frac{1}{N} \sum_{i=0}^{N-1}(x_i \cdot y_i))-( \frac{1}{N} \sum_{i=0}^{N-1}x_i )(\frac{1}{N} \sum_{i=0}^{N-1}y_i)$$

等会儿你就会看到，样本和它自身的协方差—$Cov(x,x)$—就等于它的方差。在*d*维空间中（对于一个像素的RGB来说，*d=3*），我们用协方差矩阵
$$\sideset{}{_{x,y}} \sum$$
来描述更方便一些。协方差矩阵既包括变量之间的协方差，也包括自身的方差。协方差矩阵是对称的，即
$$\sideset{}{_{x,y}} \sum =  \sideset{}{_{y,x}} \sum$$

之前我们遇到过一个函数，当我们处理一个向量时可以用到它。这个函数是`cv::calcCovarMatrix()`。该函数允许我们提供 $N$ 个 $d$ 维向量，输出结果为 $d \times d$ 维的协方差矩阵。然而，我们现在的问题是我们要为每一个像素点计算协方差矩阵（或者至少，在三维RGB图像的情况下，我们要在这个矩阵中计算6个量。）

实际上，最好的方法就是简单地使用我们之前已经建立的代码来计算方差，然后分别计算三个新的元素（$ \sideset{}{x,y} \sum$的非对角线元素）。看看协方差的形式就知道，`cv::accumulateSquare()`在这里不管用，因为我们需要计算$x_i \cdot y_i$。OpenCV中的`cv::accumulateProduct()`这个函数可以做到这点。

```c++
void accumulateProduct(
  cv::InputArray         src1,                 // 输入图像，1 通道或者 3 通道，U8 或者 F32
  cv::InputArray         src2,                 // 输入图像，1 通道或者 3 通道，U8 或者 F32
  cv::InputOutputArray   dst,                  // 结果图像， F32 或者 F64
  cv::InputArray         mask = cv::noArray    // Use ssrc pixel if mask pixel != 0
);
```
这个方程和`cv::accumulateSquare()`的工作方式一样，不同于对`src`的每个元素进行平方，这个函数将`src1`和`src2`相应的元素进行相乘。很遗憾，该函数不能允许我们从这些传入的数组中抽出单个通道。对于`src1`和`src2`多通道的情况，计算结果是在每个通道的基础上完成的。

对于我们目前想要计算的协方差模型中的非对角元素来说，这些其实并不是我们真正想要的。相反，我们要的是同一张图片中不同的通道。为了达到这个目的，我们必须用`cv::split()`把输入图像分开。如 *Example 15-3* 所示。

*Example 15-3. Computing the off-diagonal elements of a covariance model*

```c++
vector<cv::Mat> planes(3);
vector<cv::Mat> sums(3);
vector<cv::Mat> xysums(6);

int image_count = 0;

void accumulateCovariance(cv::Mat& I) {
  int i, j, n;

  if (sum.empty) {
    for( i=0; i<3; i++)
    { // the r, g, b sums
      sums[i] = cv::Mat::zeros(I.size(), CV::F32C1 );
    }
    for ( n = 0; n < 6; n++)
    { // the rr, rg, rb, gg, gb, and bb elements
      xysums[n]=cv::Mat::zeros( I.size(), CV::F32C1 );
    }
  }

  cv::split( I, rgb );
  for( i=0; i<3; i++)
  {
    cv::accumulate( rgb[i],sums[i]);
  }
  n=0;
  for ( i=0; i<3; i++)
  {
    for ( j=i; j<3; j++)
    {
      n++;
      cv::accumulateProduct( rgb[i], rgb[j], xysums[n] );
    }
  }
  image_count++;
}


```
相应的计算函数也是我们之前用来计算方差的函数的轻微变种。

```c++
// note that `variance` is sigma^2
//
void computeVariance(
  cv::Mat& covariance          // a six-channel array, channels are the rr, rg, gg, gb, and bb elements of Sigma_xy
) {
  double one_by_N = 1.0 / image_count;

  // reuse the xysum arrays as storage for individual entries
  //
  int n = 0;
  for (int i = 0; i < 3; i++) {   // "row" of Sigma
    for(int j=i; j<3; j++)
    {
      n++;
      xysums[n] = one_by_N * xysums[n]- (one_by_N * one_by_N) * sums[i].mul(sums[j]);
    }
  }

  // reassemble the six individual elements into a six-chnnel array
  //
  cv::merge( xysums, covariance );
}

```

### A brief note on model testing and cv::Mahalanobis()

在这部分，我们介绍一些稍微复杂一点的模型，但是不讨论如何测试一幅新图像中的某些点是否在背景方差所预测的范围内。在只有方差的模型（所有通道上的高斯模型有一个隐含的假设：每个通道之间是统计独立的）中，然而每个维度的方差不一定是相等的，所以这个问题就被复杂化了。而我们要说的是另一种方法，对每一个维度计算 $z-score$ （被标准差归一化的到均值的距离：$(x- \overline x)/ \sigma_x$）。然后将多维尺度的A归纳为平方和的平方根,例如：

$$ \sqrt {z_{red}^2+z_{green}^2+z_{blue}^2} $$

 在整个协方差矩阵中，

# A More Advanced Background Subtraction Method


# Connected Components for Foreground Cleanup


## A Quick Test


# Comparing Two Background Methods


# OpenCV Background Subtraction Encapsulation


## The cv::BackgroundSubtractor Base class


## KaewTraKuPong and Bowden Method


## Zivkovic Method


# Summary


# Exercises
