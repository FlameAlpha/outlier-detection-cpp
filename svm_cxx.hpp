#ifndef SVM_CXX_HPP
#define SVM_CXX_HPP

#include <string>
#include <vector>
#include <iostream>
#include "libsvm/svm.h"
#include "dataframe.hpp"

#define Malloc(type, n) (type *)malloc((n)*sizeof(type))

class svm_cxx {
private:
    struct svm_model *model;
    struct svm_parameter param{};
    struct svm_problem prob{};
    struct svm_node *x_space;
    struct svm_node *svm_node_data;
    int feature_num;
public:
    explicit svm_cxx(int _feature_num, const std::string &filename = "") :
        model(nullptr),
        x_space(nullptr),
        feature_num(_feature_num) {
        prob.l = 0;
        prob.x = nullptr;
        prob.y = nullptr;
        svm_node_data = Malloc(struct svm_node,feature_num + 1);
        if(!filename.empty())
            load_model(filename);
    }

    ~svm_cxx() {
        free_model();
        free_param();
        free_dataset();
        free(svm_node_data);
    }

    void param_init(int svm_type = C_SVC, int kernel_type = RBF, int degree = 3, double gamma = 0, double coef0 = 0,
                    double nu = 0.5, double C = 1, double eps = 1e-3, double cache_size = 200, double p = 0.1,
                    int shrinking = 1, int probability = 0, const std::vector<std::pair<int, double>> &nr_weight = {}) {
        param.svm_type = svm_type; //set type of SVM (default C_SVC)
        param.kernel_type = kernel_type; //set type of kernel function (default RBF)
        param.degree = degree; //set degree in kernel function (default 3)
        param.coef0 = coef0; //set coef0 in kernel function (default 0)
        param.gamma = gamma; //set gamma in kernel function (default 1/num_features)
        param.nu = nu; //set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
        param.C = C; //set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
        param.eps = eps; //set tolerance of termination criterion (default 0.001)
        param.cache_size = cache_size; //set cache memory size in MB (default 100)
        param.p = p; //set the epsilon in loss function of epsilon-SVR (default 0.1)
        param.shrinking = shrinking; //whether to use the shrinking heuristics, 0 or 1 (default 1)
        param.probability = probability; //whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
        param.nr_weight = nr_weight.size();
        if (param.nr_weight > 0) {
            //set the parameter C of class i to weight*C, for C-SVC (default 1)
            param.weight_label = (int *) malloc(sizeof(int) * param.nr_weight);
            param.weight = (double *) malloc(sizeof(double) * param.nr_weight);
            for (const auto &item : nr_weight) {
                param.weight_label[param.nr_weight - 1] = item.first;
                param.weight[param.nr_weight - 1] = item.second;
            }
        } else {
            param.weight = nullptr;
            param.weight_label = nullptr;
        }
    }

    void one_class_svm_param_init() {
        // -s 2 -t 2 -d 3 -g 0 -r 0 -n 0.001 -c 1 -e 0.001 -m 200 -p 0.1 -h 1 -b 0
        param_init(ONE_CLASS, RBF, 3, 0, 0, 0.0015, 1, 1e-3,200);
    }

    void read_problem(const dataframe<double> &dataset, const std::vector<double> &label){
        if (dataset.empty() || (label.empty() && param.svm_type != ONE_CLASS))
            return;

        free_dataset();

        auto len = dataset.row_num();
        auto dim = dataset.column_num();
        auto elements = len * dim;

        prob.l = len;
        prob.y = Malloc(double, prob.l);
        prob.x = Malloc(struct svm_node *, prob.l);
        x_space = Malloc(struct svm_node, elements + len);

        int j = 0;
        for (int l = 0; l < len; l++) {
            prob.x[l] = &x_space[j];
            for (int d = 0; d < dim; d++) {
                x_space[j].index = d + 1;
                x_space[j].value = dataset(d)[l];
                j++;
            }
            x_space[j++].index = -1;

            if (param.svm_type != ONE_CLASS)
                prob.y[l] = label[l];
            else prob.y[l] = 1;
        }
    }

    double train(const dataframe<double> &dataset, const std::vector<double> &label = {}, int nr_fold = 5) {
        if(dataset.column_num() != feature_num)
            return -1;
        free_model();
        read_problem(dataset, label);
        if(prob.l <= 0)
            return -1;
        if (param.gamma < 1e-6)
            param.gamma = 1.0 / double(dataset.column_num());
        auto error_log = svm_check_parameter(&prob, &param);
        if (error_log != nullptr) {
            std::cout << error_log;
            return 0;
        }
        model = svm_train(&prob, &param);
        double accaurcy;
        if(nr_fold > 1)
            accaurcy = cross_validation(nr_fold);
        else accaurcy = clf_validation(dataset, label);
        return accaurcy;
    }

    std::pair<double, double> predict(svm_node *_svm_node_data) {
        double result;
        double dec_value;
        if (svm_check_probability_model(model)) {
            auto prob_vector = Malloc(double, model->nr_class);
            result = svm_predict_probability(model, _svm_node_data, prob_vector);
            for (int k = 0; k < model->nr_class; k++) {
                if (model->label[k] == result) {
                    dec_value = prob_vector[k];
                    break;
                }
            }
            delete[] prob_vector;
        } else {
            result = svm_predict_values(model, _svm_node_data, &dec_value);
        }
        return {result, dec_value};
    }

    std::pair<double,double> predict(const std::vector<double> &data) {
        for (int i = 0; i < data.size(); i++) {
            svm_node_data[i].index = i + 1;
            svm_node_data[i].value = data[i];
        }
        svm_node_data[data.size()].index = -1;
        return std::move(predict(svm_node_data));
    }

    int load_model(const std::string &model_path) {
        free_model();
        model = svm_load_model(model_path.c_str());
        if (model == nullptr)
            return -1;
        return 0;
    }

    int save_model(const std::string &model_path) {
        return svm_save_model(model_path.data(), model);
    }

    double clf_validation(const dataframe<double> &dataset, const std::vector<double> &label = {}){
        if(((label.size() < dataset.row_num()) && (model->param.svm_type != ONE_CLASS)) ||
            dataset.column_num() != feature_num)
            return -1;
        double total_correct = 0;
        svm_node_data[dataset.column_num()].index = -1;
        for (int i = 0; i < dataset.row_num(); i++) {
            for (int j = 0; j < dataset.column_num(); j++) {
                svm_node_data[j].index = j + 1;
                svm_node_data[j].value = dataset(j)[i];
            }
            if(model->param.svm_type == ONE_CLASS) {
                if (svm_predict(model, svm_node_data) == int(1))
                    ++total_correct;
            }else{
                if (svm_predict(model, svm_node_data) == int(label[i]))
                    ++total_correct;
            }
        }
        return 100.0 * total_correct / double(dataset.row_num());
    }

    double cross_validation(int nr_fold = 5) {
        int i;
        int total_correct = 0;
        double total_error = 0;
        double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
        auto target = Malloc(double, prob.l);
        svm_cross_validation(&prob, &param, nr_fold, target);
        if (param.svm_type == EPSILON_SVR ||
            param.svm_type == NU_SVR) {
            for (i = 0; i < prob.l; i++) {
                double y = prob.y[i];
                double v = target[i];
                total_error += (v - y) * (v - y);
                sumv += v;
                sumy += y;
                sumvv += v * v;
                sumyy += y * y;
                sumvy += v * y;
            }
            printf("Cross Validation Mean squared error = %g\n", total_error / prob.l);
            printf("Cross Validation Squared correlation coefficient = %g\n",
                   ((prob.l * sumvy - sumv * sumy) * (prob.l * sumvy - sumv * sumy)) /
                   ((prob.l * sumvv - sumv * sumv) * (prob.l * sumyy - sumy * sumy))
            );
        } else {
            for (i = 0; i < prob.l; i++)
                if (int(target[i]) == int(prob.y[i]))
                    ++total_correct;
            printf("Cross Validation Accuracy = %g%%\n", 100.0 * total_correct / prob.l);
        }
        free(target);
        return 100.0 * total_correct / prob.l;
    }

private:
    void free_model() {
        svm_free_and_destroy_model(&model);
    }

    bool free_param() {
        if (param.weight != nullptr) {
            free(param.weight_label);
            param.weight_label = nullptr;
            free(param.weight);
            param.weight = nullptr;
            return true;
        }
        return false;
    }

    bool free_dataset() {
        if (prob.l != 0 && model == nullptr) {
            prob.l = 0;
            if(prob.y != nullptr) {
                free(prob.y);
                prob.y = nullptr;
            }
            if(prob.x != nullptr) {
                free(prob.x);
                prob.x = nullptr;
            }
            if(x_space != nullptr) {
                free(x_space);
                x_space = nullptr;
            }
            return true;
        }
        return false;
    }
};

#endif //SVM_CXX_HPP