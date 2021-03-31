#ifndef DETECTION_HPP
#define DETECTION_HPP

#include "svm_cxx.hpp"

template<typename data_type = double,
        template<typename> class scaler_type = standard_scaler>
class detection {
    scaler_type<data_type> user_scaler;
    svm_cxx one_class_svm;
public:
    detection(int feature_num, const std::string &model_filename, const std::string &scaler_filename) :
            user_scaler(scaler_filename),
            one_class_svm(feature_num, model_filename) {
    }

    std::pair<double,data_type> predict(std::vector<data_type> &data, bool trans = true) {
        if (trans){
            user_scaler.transform(data);
        }
        return std::move(one_class_svm.predict(data));
    }

    double validation(dataframe<data_type> &dataset, bool trans = true) {
        if (trans && !dataset.get_scaler_flag()){
            user_scaler.transform(dataset);
        }
        return one_class_svm.clf_validation(dataset);
    }

    void validation(const dataframe<data_type> &dataset, const std::string & filename, bool trans = true) {
        std::vector<std::string> column_strs = dataset.get_column_str();
        column_strs.emplace_back("result");
        column_strs.emplace_back("dec_value");
        dataframe<data_type> save_file(column_strs);
        for (int i = 0; i < dataset.row_num(); ++i) {
            std::vector<data_type> data = dataset[i].get_std_vector();
            std::vector<data_type> data_backup = data;
            std::pair<double,data_type> result = this->predict(data,trans);
            data_backup.emplace_back(result.first);
            data_backup.emplace_back(result.second);
            save_file.append(data_backup);
        }
        save_file.to_csv(filename);
    }
};

#endif //DETECTION_HPP
