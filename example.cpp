#include <random>
#include <functional>
#include "detection.hpp"

int main() {
    svm_cxx one_class_svm(2);

    std::random_device source;

    std::vector<unsigned long int> random_data(42);
    std::generate(random_data.begin(), random_data.end(), std::ref(source));
    std::seed_seq seeds(random_data.begin(), random_data.end());

    std::mt19937 gen(seeds);

    std::uniform_real_distribution<double> uniform_dist(-8, 8);
    std::normal_distribution<double> normal_dist{0,1};

    dataframe<double> train_set(2);
    dataframe<double> test_set_true(2);
    dataframe<double> test_set_false(2);

    for (int i = 0; i < 100; ++i) {
        train_set.append({0.3 * normal_dist(gen)+2,0.3 * normal_dist(gen)+2});
        train_set.append({0.3 * normal_dist(gen)-2,0.3 * normal_dist(gen)-2});
    }

    for (int i = 0; i < 20; ++i) {
        test_set_true.append({0.3 * normal_dist(gen)+2,0.3 * normal_dist(gen)+2});
        test_set_true.append({0.3 * normal_dist(gen)-2,0.3 * normal_dist(gen)-2});
        test_set_false.append({uniform_dist(gen),uniform_dist(gen)});
        test_set_false.append({uniform_dist(gen),uniform_dist(gen)});
    }

    standard_scaler<double> scaler(train_set);
    scaler.transform(train_set);
    scaler.transform(test_set_true);
    scaler.transform(test_set_false);

    one_class_svm.one_class_svm_param_init();
    double accuracy = one_class_svm.train(train_set, {}, 1);
    std::cout << "Validation accuracy of training dataset = " << accuracy << "%" << std::endl;

    if (!one_class_svm.save_model("../model/one_class_svm_cxx"))
        std::cout << "Save model successfully\n";
    if (!one_class_svm.load_model("../model/one_class_svm_cxx"))
        std::cout << "Loading model successfully\n";

    std::cout << "Validation accuracy of test true dataset = " << one_class_svm.clf_validation(test_set_true) << "%\n";
    std::cout << "Validation accuracy of test false dataset = " << 100 - one_class_svm.clf_validation(test_set_false) << "%\n";
}