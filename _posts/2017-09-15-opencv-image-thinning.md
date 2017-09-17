---
layout: post
title: Opencv之图像细化/骨架提取
date: 2017-09-15 14:15:00
categories:
- Opencv
tags:
- Opencv
---

表示一个平面区域的结构形状的一种重要方法是将它简化为图形。这种简化可以通过一种细化（也称为骨架化）算法得到该区域的骨架来实现。

## Image thinning

在大范围的图像处理问题中，细化过程起着核心作用，从印刷电路板的自动检测到空气过滤器中的石棉纤维的计数，再到三维测量中激光光条中心点的提取。
然而Opencv-3.1版本以及之前版本都没有包含图像细化（骨架提取）之类的算法。直到3.2版本发布时，才有了图像细化（骨架提取）的算法，放在了所谓的未稳定功能模块opencv_contrib中。只安装了opencv正式版本的童鞋可能要多动动手，从github上下载完整的opencv版本(opencv-master)+(opencv_contrib-master)，通过Cmake编译安装。具体过程请自行百度。不会百度的童鞋请看这本书[opencv-3-computer-vision-application-programming-cookbook-third-edition](http://pan.baidu.com/s/1eSlHniE)。

这是一本很好的opencv教程，怎么安装完整版本的opencv里面讲的非常清楚。祝成功！（跑题了。。。。。。。。。。）

## thinning()

```c++
void cv::ximgproc::thinning(InputArray  src,
                            Outputarray dst,
                            int         thinningType=THINNING_ZHANGSUEN
                           )
```
