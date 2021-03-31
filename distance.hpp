#ifndef DISTANCE_HPP
#define DISTANCE_HPP

#include <cmath>
#include <tuple>
#include <vector>
#include <limits>
#include <iostream>
#include "dataframe.hpp"

template<typename T = double>
class angle2radian {
public:
    T operator()(T angle) {
        return angle / 180.0f * 3.1415926f;
    }
};

template<typename T = double>
class [[maybe_unused]] radian2angle {
public:
    T operator()(T radian) {
        return radian / 3.1415926f * 180.0f;
    }
};

enum gyroscope {
    roll = 1,
    pitch,
    course
};

template<typename T = double>
class pose {
    T roll_angle;
    T pitch_angle;
    T course_angle;
    T error{};
public:
    pose(T _roll, T _pitch, T _course) : roll_angle(_roll), pitch_angle(_pitch), course_angle(_course) {};

    static gyroscope reverse_map(gyroscope index) {
        if (index == pitch)
            return roll;
        else return pitch;
    }

    T &operator[](gyroscope index) {
        if (index == roll) {
            return roll_angle;
        } else if (index == pitch) {
            return pitch_angle;
        } else if (index == course) {
            return course_angle;
        } else return error;
    }

    T &operator[](unsigned short index) {
        if (index == 0) {
            return roll_angle;
        } else if (index == 1) {
            return pitch_angle;
        } else if (index == 2) {
            return course_angle;
        } else return error;
    }

    const T &operator[](gyroscope index) const {
        if (index == roll) {
            return roll_angle;
        } else if (index == pitch) {
            return pitch_angle;
        } else if (index == course) {
            return course_angle;
        } else return error;
    }

    const T &operator[](unsigned short index) const {
        if (index == 0) {
            return roll_angle;
        } else if (index == 1) {
            return pitch_angle;
        } else if (index == 2) {
            return course_angle;
        } else return error;
    }
};

class info_user {
public:
    // up to down
    float length_chest = 1;
    float length_shank = 1;
    float length_thigh = 1;
};

enum angle_index {
    angle_shank_l = 0,
    angle_shank_r,
    angle_thigh_l,
    angle_thigh_r,
    angle_chest
};

template<typename T = double, template<typename> class transform = angle2radian>
class vertical_offset {
public:
    T operator()(const pose<T> &angle, const gyroscope &index = pitch, const T distance = 1,
                 transform<T> t = transform<T>()) {
        return std::sin(t(angle[index])) * distance;
    }
};

template<typename T = double, template<typename> class transform = angle2radian>
class horizontal_offset {
public:
    T operator()(const pose<T> &angle, const gyroscope &index = pitch, const T distance = 1,
                 transform<T> t = transform<T>()) {
        return std::sin(t(angle[pose<T>::reverse_map(index)])) * std::cos(t(angle[index])) * distance;
    }
};

template<template<typename, template<typename> class transform = angle2radian> class function = vertical_offset, typename T = double>
T get_offset(const std::vector<pose<T>> &angles,
                 const std::vector<gyroscope> &indexes,
                 const std::vector<T> &distances,
                 function<T> f = function<T>()) {
    if (angles.size() == distances.size() && angles.size() == indexes.size()) {
        T count = 0;
        for (unsigned int i = 0; i < angles.size(); i++) {
            count += f(angles[i], indexes[i], distances[i]);
        }
        return count;
    } else return 0;
}

template<template<typename, template<typename> class transform = angle2radian> class function = vertical_offset, typename T = double>
std::vector<T> get_joint_offset(const std::vector<pose<T>> &angles,
             const std::vector<gyroscope> &indexes,
             const std::vector<T> &distances,
             function<T> f = function<T>()) {
    if (angles.size() == distances.size() && angles.size() == indexes.size()) {
        std::vector<T> data_list;
        for (unsigned int i = 0; i < angles.size(); i++) {
            data_list.emplace_back(f(angles[i], indexes[i], distances[i]));
        }
        return std::move(data_list);
    } else return {};
}

template<template<typename, template<typename> class transform = angle2radian> class function = vertical_offset, typename T = double>
std::vector<T>
get_multi_offset(const std::vector<std::tuple<std::vector<pose<T>>, std::vector<gyroscope>, std::vector<T>>> &data) {
    std::vector<T> position;
    position.reserve(data.size());
    for (const auto &[angle, index, distance] : data) {
        position.emplace_back(get_offset<function,T>(angle, index, distance));
    }
    return std::move(position);
}

template<template<typename, template<typename> class transform = angle2radian> class function = vertical_offset, typename T = double>
std::vector<T>
get_multi_joint_offset(const std::vector<std::tuple<std::vector<pose<T>>, std::vector<gyroscope>, std::vector<T>>> &data) {
    std::vector<T> position;
    position.reserve(data.size());
    for (const auto &[angle, index, distance] : data) {
        for (const auto & item : get_joint_offset<function,T>(angle, index, distance))
            position.emplace_back(item);
    }
    return std::move(position);
}

template<typename data_type, typename T = double>
std::vector<T>
get_axis_offset(std::vector<data_type> data, std::pair<gyroscope, gyroscope> select = {roll, pitch}) {
    info_user info;
    const std::vector<std::tuple<std::vector<pose<T>>, std::vector<gyroscope>, std::vector<T>>> &single_data = {
            {{pose<T>(data[angle_chest * 3].imudata.r[0], -data[angle_chest * 3 + 1].imudata.r[1], data[angle_chest * 3 + 2].imudata.r[2])}, {select.first},{info.length_chest}},
            {{pose<T>(data[angle_shank_l * 3].imudata.r[0], data[angle_shank_l * 3 + 1].imudata.r[1], data[angle_shank_l * 3 + 2].imudata.r[2]),    pose<T>(data[angle_thigh_l * 3].imudata.r[0], data[angle_thigh_l * 3 + 1].imudata.r[1], data[angle_thigh_l * 3 + 2].imudata.r[2])},    {select.second, select.second}, {info.length_shank, info.length_thigh}},
            {{pose<T>(-data[angle_shank_r * 3].imudata.r[0], -data[angle_shank_r * 3 + 1].imudata.r[1], -data[angle_shank_r * 3 + 2].imudata.r[2]), pose<T>(-data[angle_thigh_r * 3].imudata.r[0], -data[angle_thigh_r * 3 + 1].imudata.r[1], -data[angle_thigh_r * 3 + 2].imudata.r[2])}, {select.second, select.second}, {info.length_shank, info.length_thigh}}};
    auto position = get_multi_offset<vertical_offset, T>(single_data);
    for (const auto &temp : get_multi_offset<horizontal_offset, T>(single_data))
        position.push_back(temp);
    return position;
}

#define get_offset false

template<typename T = double>
std::pair<T, T> get_multi_axis_offset(const dataframe<T> &dataset, dataframe<T> &all_position,
                                              std::pair<gyroscope, gyroscope> select = {gyroscope::roll, gyroscope::pitch}) {
    info_user info;
#if get_offset
    float max_distance = std::numeric_limits<T>::min();
    float min_distance = std::numeric_limits<T>::max();
#endif
    for (int index = 0; index < dataset.row_num(); index++) {
        const std::vector<std::tuple<std::vector<pose<T>>, std::vector<gyroscope>, std::vector<T>>> &data ={
                {{pose<T>(dataset(angle_chest * 3)[index], -dataset(angle_chest * 3 + 1)[index], dataset(angle_chest * 3 + 2)[index])}, {select.first}, {info.length_chest}},
                {{pose<T>(dataset(angle_shank_l * 3)[index], dataset(angle_shank_l * 3 + 1)[index], dataset(angle_shank_l * 3 + 2)[index]), pose<T>(dataset(angle_thigh_l * 3)[index], dataset(angle_thigh_l * 3 + 1)[index], dataset(angle_thigh_l * 3 + 2)[index])}, {select.second, select.second}, {info.length_shank, info.length_thigh}},
                {{pose<T>(-dataset(angle_shank_r * 3)[index], -dataset(angle_shank_r * 3 + 1)[index], -dataset(angle_shank_r * 3 + 2)[index]), pose<T>(-dataset(angle_thigh_r * 3)[index], -dataset(angle_thigh_r * 3 + 1)[index], -dataset(angle_thigh_r * 3 + 2)[index])}, {select.second, select.second}, {info.length_shank, info.length_thigh}}};

        auto position1 = get_multi_offset<vertical_offset,T>(data);
#if get_offset
        float distance1 = position1[0] - (position1[1] + position1[2]) / 2.;
        position1.push_back(std::abs(distance1));
#endif
        auto position2 = get_multi_offset<horizontal_offset,T>(data);
#if get_offset
        float distance2 = position2[0] - (position2[1] + position2[2]) / 2.;
        position2.push_back(std::abs(distance2));
        float distance = distance1 * distance1 + distance2 * distance2;
        if (distance > max_distance) {
            max_distance = distance;
        } else if (distance < min_distance) {
            min_distance = distance;
        }
#endif
        for (const auto &temp : position2)
            position1.push_back(temp);

#if get_offset
        position1.emplace_back(distance);
#endif
        all_position.append(position1);
    }
#if get_offset
    return {min_distance, max_distance};
#else
    return {0, 0};
#endif
}

template<typename T = double>
std::pair<T, T> get_multi_axis_joint_offset(const dataframe<T> &dataset, dataframe<T> &all_position,
                                      std::pair<gyroscope, gyroscope> select = {gyroscope::roll, gyroscope::pitch}) {
    info_user info;
#if get_offset
    float max_distance = std::numeric_limits<T>::min();
    float min_distance = std::numeric_limits<T>::max();
#endif
    for (int index = 0; index < dataset.row_num(); index++) {
        const std::vector<std::tuple<std::vector<pose<T>>, std::vector<gyroscope>, std::vector<T>>> &data ={
                {{pose<T>(dataset(angle_chest * 3)[index], -dataset(angle_chest * 3 + 1)[index], dataset(angle_chest * 3 + 2)[index])}, {select.first}, {info.length_chest}},
                {{pose<T>(dataset(angle_shank_l * 3)[index], dataset(angle_shank_l * 3 + 1)[index], dataset(angle_shank_l * 3 + 2)[index]), pose<T>(dataset(angle_thigh_l * 3)[index], dataset(angle_thigh_l * 3 + 1)[index], dataset(angle_thigh_l * 3 + 2)[index])}, {select.second, select.second}, {info.length_shank, info.length_thigh}},
                {{pose<T>(-dataset(angle_shank_r * 3)[index], -dataset(angle_shank_r * 3 + 1)[index], -dataset(angle_shank_r * 3 + 2)[index]), pose<T>(-dataset(angle_thigh_r * 3)[index], -dataset(angle_thigh_r * 3 + 1)[index], -dataset(angle_thigh_r * 3 + 2)[index])}, {select.second, select.second}, {info.length_shank, info.length_thigh}}};

        auto position1 = get_multi_joint_offset<vertical_offset,T>(data);
#if get_offset
        float distance1 = position1[0] - (position1[1] + position1[2]) / 2.;
        position1.push_back(std::abs(distance1));
#endif
        auto position2 = get_multi_joint_offset<horizontal_offset,T>(data);
#if get_offset
        float distance2 = position2[0] - (position2[1] + position2[2]) / 2.;
        position2.push_back(std::abs(distance2));
        float distance = distance1 * distance1 + distance2 * distance2;
        if (distance > max_distance) {
            max_distance = distance;
        } else if (distance < min_distance) {
            min_distance = distance;
        }
#endif
        for (const auto &temp : position2)
            position1.push_back(temp);

#if get_offset
        position1.emplace_back(distance);
#endif
        all_position.append(position1);
    }
#if get_offset
    return {min_distance, max_distance};
#else
    return {0, 0};
#endif
}

template <typename T = double>
void get_position(const std::vector<std::string> &filenames, dataframe<T> &all_position,
                  const std::string &prefix = "") {
    T max_distance = std::numeric_limits<T>::min();
    T min_distance = std::numeric_limits<T>::max();
    for (const auto &filename : filenames) {
        auto[_min_distance, _max_distance] = get_multi_axis_offset(dataframe<T>(prefix + filename), all_position);
        if (_max_distance > max_distance) {
            max_distance = _max_distance;
        }
        if (_min_distance < min_distance) {
            min_distance = _min_distance;
        }
    }
#if get_offset
    std::cout << "min distance is : " << min_distance << " max distance is : " << max_distance << std::endl;
#endif
}

template <typename T = double>
void get_joint_position(const std::vector<std::string> &filenames, dataframe<T> &all_position,
                        const std::string &prefix = "") {
    T max_distance = std::numeric_limits<T>::min();
    T min_distance = std::numeric_limits<T>::max();
    for (const auto &filename : filenames) {
        auto[_min_distance, _max_distance] = get_multi_axis_joint_offset(dataframe<T>(prefix + filename), all_position);
        if (_max_distance > max_distance) {
            max_distance = _max_distance;
        }
        if (_min_distance < min_distance) {
            min_distance = _min_distance;
        }
    }
#if get_offset
    std::cout << "min distance is : " << min_distance << " max distance is : " << max_distance << std::endl;
#endif
}

#endif //DISTANCE_HPP
