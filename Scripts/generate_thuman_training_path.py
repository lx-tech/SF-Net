# -*coding: utf-8 -*-
import os


# define sone struct
ROOT_PATH = 'G:/lx/Datasets/thuman2.0/'  # root path

# the file's path and format
# RAW_DATA_FOLDER = 'Kitti2012/training/%s/'
RAW_DATA_FOLDER = 'training/%s/'
RGB_FOLDER = 'color'
DEPTH_FOLDER = 'depth'
RGB_LABLE_FOLDER = 'color_gt'
DEPTH_LABLE_FOLDER = 'depth_gt'
FILE_NAME = '%04d_0_00'

# file type
RAW_COLOR_TYPE = '.jpg'
RAW_DEPTH_TYPE = '.png'

# the output's path,
# TRAIN_LIST_PATH = './Datasets/kitti2012_training_list.csv'
# VAL_TRAINLIST_PATH = './Datasets/kitti2012_val_list.csv'
TRAIN_LIST_PATH = './Datasets/thuman_training_list.csv'
VAL_TRAINLIST_PATH = './Datasets/thuman_testing_val_list.csv'

# IMG_NUM = 194  # the dataset's total image
IMG_NUM = 60    # the dataset's total image
TIMES = 5      # the sample of val

TEST_FLAG = True


def gen_color_path(file_folder: str, num: int) -> str:
    path = ROOT_PATH + RAW_DATA_FOLDER % file_folder + FILE_NAME % num + \
        RAW_COLOR_TYPE
    return path

def gen_depth_path(file_folder: str, num: int) -> str:
    path = ROOT_PATH + RAW_DATA_FOLDER % file_folder + FILE_NAME % num + \
        RAW_DEPTH_TYPE
    return path

def open_file() -> object:
    if os.path.exists(TRAIN_LIST_PATH):
        os.remove(TRAIN_LIST_PATH)
    if os.path.exists(VAL_TRAINLIST_PATH):
        os.remove(VAL_TRAINLIST_PATH)

    fd_train_list = open(TRAIN_LIST_PATH, 'a')
    fd_val_train_list = open(VAL_TRAINLIST_PATH, 'a')

    data_str = "color_img,depth_img,color_gt,depth_gt"
    output_data(fd_train_list, data_str)
    output_data(fd_val_train_list, data_str)

    return fd_train_list, fd_val_train_list


def output_data(output_file: object, data: str) -> None:
    output_file.write(str(data) + '\n')
    output_file.flush()


def gen_list(fd_train_list, fd_val_train_list):
    total = 0
    off_set = 1
    for num in (list(range(1,31))+list(range(330,360))):

        color_path = gen_color_path(RGB_FOLDER, num)
        color_lable_path = gen_color_path(RGB_LABLE_FOLDER, 0)
        depth_path = gen_depth_path(DEPTH_FOLDER, num)
        depth_lable_path = gen_depth_path(DEPTH_LABLE_FOLDER, 0)

        color_path_is_exists = os.path.exists(color_path)
        color_lable_path_is_exists = os.path.exists(color_lable_path)
        depth_path_is_exists = os.path.exists(depth_path)
        depth_lable_path_is_exists = os.path.exists(depth_lable_path)

        if (not color_path_is_exists) and (not color_lable_path_is_exists)\
                (not depth_path_is_exists) and (not depth_lable_path_is_exists):
           break

        if (off_set + num) % TIMES == 0:
            data_str = color_path + ',' + depth_path + ',' + color_lable_path + ',' + depth_lable_path
            output_data(fd_val_train_list, data_str)
        else:
            data_str = color_path + ',' + depth_path + ',' + color_lable_path + ',' + depth_lable_path
            output_data(fd_train_list, data_str)

        total = total + 1

    return total


def main() -> None:
    fd_train_list, fd_val_train_list = open_file()
    total = gen_list(fd_train_list, fd_val_train_list)
    print(total)


if __name__ == '__main__':
    main()
