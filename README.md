# outlier-detection-cpp

#### 介绍
本项目基于 [libsvm-cpp](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) 以及 [dataframe-cpp](https://gitee.com/flamealpha/dataframe-cpp) 进行开发，主要用于异常检测，可直接读取CSV文件进行训练，存储和读取model以及scaler，并用于在线异常检测。
同时该项目使 [libsvm-cpp](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) 支持直接读取CSV文件进行训练和测试。

#### 示例代码

```cpp
#include <random>
#include <functional>
#include "detection.hpp"

int main() {
    // 创建一个SVM分类器
    svm_cxx one_class_svm(2);
    
    // 创建使用硬件熵源的非确定随机数生成器
    std::random_device source;
    // 生成随机种子序列
    std::vector<unsigned long int> random_data(42);
    std::generate(random_data.begin(), random_data.end(), std::ref(source));
    std::seed_seq seeds(random_data.begin(), random_data.end());

    // 创建 32 位梅森缠绕器(随机数生产器)
    std::mt19937 gen(seeds);

    // 创建均匀分布对象 和 正态分布对象
    std::uniform_real_distribution<double> uniform_dist(-8, 8);
    std::normal_distribution<double> normal_dist{0,1};

    // 创建三个数据库分别用于训练模型，测试正样本准确率，测试负样本准确率
    dataframe<double> train_set(2);
    dataframe<double> test_set_true(2);
    dataframe<double> test_set_false(2);

    // 为训练数据集添加数据
    for (int i = 0; i < 100; ++i) {
        train_set.append({0.3 * normal_dist(gen)+2,0.3 * normal_dist(gen)+2});
        train_set.append({0.3 * normal_dist(gen)-2,0.3 * normal_dist(gen)-2});
    }

    // 为测试数据集添加数据
    for (int i = 0; i < 20; ++i) {
        test_set_true.append({0.3 * normal_dist(gen)+2,0.3 * normal_dist(gen)+2});
        test_set_true.append({0.3 * normal_dist(gen)-2,0.3 * normal_dist(gen)-2});
        test_set_false.append({uniform_dist(gen),uniform_dist(gen)});
        test_set_false.append({uniform_dist(gen),uniform_dist(gen)});
    }

    // 创建标准化对象 -- 同时提供了利用最大最小值归一化接口
    standard_scaler<double> scaler(train_set);
    
    // 标准化数据集
    scaler.transform(train_set);
    scaler.transform(test_set_true);
    scaler.transform(test_set_false);

    // 初始化单类分类器参数 -- 可以手动初始化，这里使用默认参数初始化
    one_class_svm.one_class_svm_param_init();
    
    // 训练单类分类器
    double accuracy = one_class_svm.train(train_set, {}, 1);
    std::cout << "Validation accuracy of training dataset = " << accuracy << "%" << std::endl;

    // 保存训练好的单类分类器模型
    if (!one_class_svm.save_model("../model/one_class_svm_cxx"))
        std::cout << "Save model successfully\n";

    // 加载已有的单类分类器模型
    if (!one_class_svm.load_model("../model/one_class_svm_cxx"))
        std::cout << "Loading model successfully\n";

    // 测试模型准确率
    std::cout << "Validation accuracy of test true dataset = " << one_class_svm.clf_validation(test_set_true) << "%\n";
    std::cout << "Validation accuracy of test false dataset = " << 100 - one_class_svm.clf_validation(test_set_false) << "%\n";
}
```

#### 最终的打印信息如下

```shell
*
optimization finished, #iter = 11
obj = 0.019323, rho = 0.128785
nSV = 5, nBSV = 0
Validation accuracy of training dataset = 97.5%
Save model successfully
Loading model successfully
Validation accuracy of test true dataset = 100%
Validation accuracy of test false dataset = 97.5%
```