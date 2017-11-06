---
layout: post
title: 【Learning Opencv3】之背景/前景分割方法总结
date: 2017-11-06 12:00:00
category: Opencv
tags:
- Opencv3
- 图像处理
---

# OpenCV3.3.1

## BackgroundSubtractorMOG2
### createBackgroundSubtractorMOG2
```c++
Ptr<BackgroundSubtractorMOG2> cv::createBackgroundSubtractorMOG2(
                                   int    history = 500,
                                   double varThreshold = 16,
                                   bool   detectShadows=true
                                 )
```
创建MOG2背景差分算子

**参数**
+ **history**
影响背景模型的历史帧的数量，历史帧数的长度,即反应背景更新速度
+ **varThreshold**
像素与模型之间的平方马氏距离的阈值，以决定像素是否被背景模型描述得很好.这个参数不会影响背景更新速度.
+ **detectShadows**
如果为真，算法将检测出阴影. 这可能会稍微影响程序运行速度，如果不需要此功能，可以把该参数设置为`false`.


## BackgroundSubtractorKNN
### createBackgroundSubtractorKNN
```c++
Ptr<BackgroundSubtractorKNN> cv::createBackgroundSubtractorKNN(
                                  int   history = 500,
                                  double dist2Threshold=400.0
                                  bool detectShadows=true
)
```
创建KNN背景差分算子.

**参数**
+ **history**
历史帧数的长度，即背景更新速率

+ **dist2Threshold**
当前像素和样本之间的平方距离，以决定当前像素是否接近样本.这个参数不会影响背景更新速率.

+ **detectShadows**
同上.
# OpenCV3.3.1_contrib

## BackgroundSubtractorMOG

## BackgroundSubtractorGMG

## BackgroundSubtractorCNT

Background subtraction based on counting.
