#include "distance.hpp"
#include "detection.hpp"

#define get_model 0x01
#define valid_model 0x02
#define get_data 0x04
#define normal_model (get_data | get_model | valid_model)

#define get_joint_model 0x10
#define valid_joint_model 0x20
#define get_joint_data 0x40
#define joint_model (get_joint_data | get_joint_model | valid_joint_model)

#define model joint_model

int main() {

#if get_joint_data & model
    dataframe<double> position({"vertical_c", "vertical_l_shank", "vertical_l_thigh",
                                "vertical_r_shank" , "vertical_r_thigh", "horizontal_c",
                                "horizontal_l_shank", "horizontal_l_thigh", "horizontal_r_shank", "horizontal_r_thigh"});
    get_joint_position({"downstarits1.csv", "levelwalk1.csv", "levelwalk2.csv", "slope1.csv",
                        "stairs1.csv", "stairs2.csv", "stand1.csv"}, position, "../dataset/");
    position.to_csv("../dataset/joint_position.csv");
    dataframe<double> position_abnormal({"vertical_c", "vertical_l_shank", "vertical_l_thigh",
                                         "vertical_r_shank" , "vertical_r_thigh", "horizontal_c",
                                         "horizontal_l_shank", "horizontal_l_thigh", "horizontal_r_shank", "horizontal_r_thigh"});
    get_joint_position({"abnormal1.csv", "abnormal2.csv", "abnormal3.csv"}, position_abnormal, "../dataset/");
    position_abnormal.to_csv("../dataset/joint_position_abnormal.csv");
#endif

#if get_joint_model & model
    svm_cxx one_class_svm(10);
    dataframe<double> train_set("../dataset/joint_position.csv");
    dataframe<double> test_set("../dataset/joint_position_abnormal.csv");
    standard_scaler<double> scaler(train_set);
    scaler.transform(train_set);
    scaler.transform(test_set);
    scaler.save_scaler("../model/joint_scaler");
    one_class_svm.one_class_svm_param_init();
    double accuracy = one_class_svm.train(train_set, {}, 1);
    std::cout << "Validation accuracy of training dataset = " << accuracy << "%" << std::endl;
    if(!one_class_svm.save_model("../model/joint_one_class_svm_cxx"))
        std::cout << "Save model successfully\n";
    if(!one_class_svm.load_model("../model/joint_one_class_svm_cxx"))
        std::cout << "Loading model successfully\n";
    std::cout << "Validation Accuracy of test dataset = " << one_class_svm.clf_validation(test_set) << "%\n";
#endif

#if valid_joint_model & model
    detection<double, standard_scaler> detector(10, "../model/joint_one_class_svm_cxx", "../model/joint_scaler");
    auto normal_dataset = dataframe<double>("../dataset/joint_position.csv");
    auto abnormal_dataset = dataframe<double>("../dataset/joint_position_abnormal.csv");
    detector.validation(normal_dataset + abnormal_dataset, "../dataset/joint_result.csv", true);
    std::cout << "Validation Accuracy of normal dataset = " << detector.validation(normal_dataset, true) << "%\n";
    std::cout << "Validation Accuracy of abnormal dataset = " << detector.validation(abnormal_dataset, true) << "%\n";
#endif

#if get_data & model
    dataframe<double> position({"vertical_c", "vertical_l", "vertical_r", "horizontal_c", "horizontal_l", "horizontal_r"});
    get_position({"downstarits1.csv", "levelwalk1.csv", "levelwalk2.csv", "slope1.csv",
                      "stairs1.csv", "stairs2.csv", "stand1.csv"}, position,"../dataset/");
    position.to_csv("../dataset/position.csv");
    dataframe<double> position_abnormal({"vertical_c", "vertical_l", "vertical_r", "horizontal_c", "horizontal_l", "horizontal_r"});
    get_position({"abnormal1.csv","abnormal2.csv","abnormal3.csv"}, position_abnormal,"../dataset/");
    position_abnormal.to_csv("../dataset/position_abnormal.csv");
#endif

#if get_model & model
    svm_cxx one_class_svm(6);
    dataframe<double> train_set("../dataset/position.csv");
    dataframe<double> test_set("../dataset/position_abnormal.csv");
    standard_scaler<double> scaler(train_set);
    scaler.transform(train_set);
    scaler.transform(test_set);
    scaler.save_scaler("../model/scaler");
    one_class_svm.one_class_svm_param_init();
    double accuracy = one_class_svm.train(train_set,{},1);
    std::cout << "Validation accuracy of training dataset = " << accuracy << "%" << std::endl;
    if(!one_class_svm.save_model("../model/one_class_svm_cxx"))
        std::cout << "Save model successfully\n";
    if(!one_class_svm.load_model("../model/one_class_svm_cxx"))
        std::cout << "Loading model successfully\n";
    std::cout << "Validation Accuracy of test dataset = " << one_class_svm.clf_validation(test_set) << "%\n";
#endif

#if valid_model & model
    detection<double, standard_scaler> detector(6, "../model/one_class_svm_cxx", "../model/scaler");
    bool write_to_file = true;
    auto normal_dataset = dataframe<double>("../dataset/position.csv");
    auto abnormal_dataset = dataframe<double>("../dataset/position_abnormal.csv");
    detector.validation(normal_dataset + abnormal_dataset, "../dataset/result.csv", true);
    std::cout << "Validation Accuracy of normal dataset = " << detector.validation(normal_dataset, true) << "%\n";
    std::cout << "Validation Accuracy of abnormal dataset = " << detector.validation(abnormal_dataset, true) << "%\n";
#endif
    return 0;
}