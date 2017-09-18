---
layout: post
title: 【Google C++ 风格指南】之命名
date: 2017-09-18 22:01:00
categories:
- CodeStyle
tags:
- CodeStyle
---

# 命名

最重要的一致性规则是命名管理。命名风格使得我们无需去查找类型申明而快速地了解某个名字所代表的含义：类型、变量、函数、常量、宏等等。我们大脑的模式匹配机制非常依赖这些命名规则。

当然，命名规则是一个很随意的事情，但是在这个领域（应该指IT领域），我们认为一致性要比个人喜好更重要。所以无论你觉得它是否合理，规则就是规则。
<!--more-->
## 通用命名规则

首先，一个名称应该是描述性的(descriptive)；要尽量避免缩写(abbreviation)。

没有任何理由，请尽可能地给出具有描述性的名称。**不要**想着去节省行空间(horizontal space)，毕竟能让新的读者快速理解你的代码才是最重要的。**不要**使用缩写！！！因为对于非项目开发者来说，那些缩写简直就是狗屎(ambiguous and unfamiliar)。**也不要**通过删掉几个字母来缩写某些单词，~~除非这个单词缩写是业界广泛认可的，比如变量i和j,一般都用在循环体中表示循环次数，num经常用来计数等等~~

```C++
int price_count_reader;  // 没有缩写
int num_errors;          //"num"是一个广泛使用的国际惯例
int num_dns_connections; //多数人都知道"DNS"是什么意思
```

```C++
int n;                 //毫无意义
int nerr;              //你告诉我这是啥！
int n_comp_conns;      //这又是啥！
int wgc_connections;   //只有你的队友知道这是啥。。。
int pc_reader;         //很多东西都可以被缩写成"pc"，
int cstmr_id;          //删除了内部的一些字母，我猜应该是是customer。哼！我就不猜！
```

**注意：**使用一些特定的国际通用的缩写是可以的，比如 `i` 用来表示一个迭代变量，`T`表示一个模板参数。

模板参数应该遵循相应的命名风格：类型模板参数应该遵循[type names](https://google.github.io/styleguide/cppguide.html#Type_Names) 的规则，而非类型模板参数应该遵循[variable names](https://google.github.io/styleguide/cppguide.html#Variable_Names)的规则。

## 文件命名
