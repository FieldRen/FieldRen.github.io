---
layout: post
date: 2017-09-18 14:05:00
catagraies: Opencv
tags: Opencv
---
OPencv学习笔记
<!--more-->

# connectedComponents
## connectedComponentsWithStats()
### 调用格式
```C++
int cv::connectedComponentsWithStats(
  InputArray image,
  OutputArray labels,
  OutputArray stats,
  OutputArray centroids,
  int connectivity,
  int ltype,
  int ccltype
)
```
### 功能描述
计算被标记的二值图像的连通区域，同时返回每一个标签的统计输出。

4连通或者8连通的图像——返回N，即从标签0到N-1的总数，其中0代表背景的标签。ltype 指定了输出标签图像的类型，这是基于标签总数或者源图像中总像素数的重要考量。ccltype指定标记联通区域所用的算法，目前支持Grana的BBDT算法和Wu的SAUF算法。

### 参数说明

**Parameters**

+ **imag:** 8-bit单通道图像

+ **labels:** 被标记的目标图像

+ **stats:** 每个标签的统计输出，包括背景标签。统计信息通过以下格式进行访问:stats(label,COLUMN)，其中COLUMN包括：

Enumerator | 描述
:------|:------
CC_STAT_LEFT | 连通区域最左边的像素的坐标  
CC_STAT_TOP | 连通区域最上边的像素的坐标
CC_STAT_WIDTH | 包围连通区域的长方形框的水平尺寸
CC_STAT_HEIGHT | 包围连通区域的长方形框的竖直尺寸
CC_STAT_AREA | 联通区域的面积
CC_STAT_MAX |

+ **centroids:** 连通区域的中心，对于每一个标签，都会输出centroids值，包括背景标签。访问方法：centroids(label,0)表示中心点的x坐标，centroids(label,1)表示中心点的y坐标。数据类型为：CV_64F。

+ **connectivity:** 8或4分别代表8连通和4连通

+ **ltype:** lable type. 输出图像标签类型。目前支持CV_32S和CV_16U。

+ **ccltype:** 连通区域算法类型，包括：

Enumerator | 描述
:------|:--------
CCL_WU      | SAUF algorithm for 8-way connectivity, SAUF algorithm for 4-way connectivity.
CCL_DEFAULT | BBDT algortihm for 8-way connectivity, SAUF algorithm for 4-way connectivity.
CCL_GRANA | BBDT algorithm for 8-way connectivity, SAUF algorithm for 4-way connectivity.
