# FireNet-Pytorch

对FireNet: A Specialized Lightweight Fire & Smoke Detection Model for Real-Time IoT Applications
Arpit（2019）的复现。

- 本repo是专门做火灾烟雾检测的，近期会开放一批数据集标准。

### 相关算法和开源模型

| name                                                       | 开源与否                                          | 数据集                                                       | 结果                                           | 整理time  |
| ---------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------- | --------- |
| FireNet(2019)                                              | Y /tf框架实现                                     |                                                              | 验证发现，效果一般；cifar10上acc=66左右。                           | 5.09/2020 |
| BoWFire(2015)                                              | N                                                 |                                                              |                                                |           |
| Fire SSD                                                   | N                                                 |                                                              |                                                | 5.15/2020 |
| RISE Video Dataset: Recognizing Industrial Smoke Emissions | 2020年5.15号刚开源 Y/基于I3D的视频分类（Pytorch） | https://github.com/CMU-CREATE-Lab/video-labeling-tool（数据集整的很复杂，各种api） | 贡献只是提出了一个监控下的工厂烟雾排放数据集； | 5.20/2020 |
| ...                                                        |                                                   |                                                              |                                                |           |