# -*coding: utf-8 -*-
import os


# define sone struct
ROOT_PATH = '/home/lixing/Documents/Thuman/thuman2.0_v2/'  # root path

# the file's path and format
RAW_DATA_FOLDER = 'training/'
RGB_FOLDER = 'color/%s/'
DEPTH_FOLDER = 'depth/%s/'
UV_FOLDER = 'uv/%s/'
RGB_LABLE_FOLDER = 'color_gt/%s/'
DEPTH_LABLE_FOLDER = 'depth_gt/%s/'
FILE_NAME = '%04d_0_00'
UV_FILE_NAME = '%04d_0_00_dp.0001'

# file type
RAW_COLOR_TYPE = '.jpg'
RAW_DEPTH_TYPE = '.png'

# the output's path,
TRAIN_LIST_PATH = './Datasets/thuman_training_list_4_30.csv'
VAL_TRAINLIST_PATH = './Datasets/thuman_testing_val_list_4_30.csv'

TIMES = 100      # the sample of val


def gen_color_path(file_folder: str, num: int) -> str:
    path = ROOT_PATH + RAW_DATA_FOLDER + RGB_FOLDER % file_folder + FILE_NAME % num + \
        RAW_COLOR_TYPE
    return path

def gen_color_gt_path(file_folder: str, num: int) -> str:
    path = ROOT_PATH + RAW_DATA_FOLDER + RGB_LABLE_FOLDER % file_folder + FILE_NAME % num + \
        RAW_COLOR_TYPE
    return path

def gen_depth_path(file_folder: str, num: int) -> str:
    path = ROOT_PATH + RAW_DATA_FOLDER +  DEPTH_FOLDER % file_folder + FILE_NAME % num + \
        RAW_DEPTH_TYPE
    return path

def gen_depth_gt_path(file_folder: str, num: int) -> str:
    path = ROOT_PATH + RAW_DATA_FOLDER +  DEPTH_LABLE_FOLDER % file_folder + FILE_NAME % num + \
        RAW_DEPTH_TYPE
    return path

def gen_uv_path(file_folder: str, num: int) -> str:
    path = ROOT_PATH + RAW_DATA_FOLDER + UV_FOLDER % file_folder + UV_FILE_NAME % num + \
        RAW_DEPTH_TYPE
    return path

def open_file() -> object:
    if os.path.exists(TRAIN_LIST_PATH):
        os.remove(TRAIN_LIST_PATH)
    if os.path.exists(VAL_TRAINLIST_PATH):
        os.remove(VAL_TRAINLIST_PATH)

    fd_train_list = open(TRAIN_LIST_PATH, 'a')
    fd_val_train_list = open(VAL_TRAINLIST_PATH, 'a')

    data_str = "color_img,depth_img,uv_img,color_gt,depth_gt"
    output_data(fd_train_list, data_str)
    output_data(fd_val_train_list, data_str)

    return fd_train_list, fd_val_train_list


def output_data(output_file: object, data: str) -> None:
    output_file.write(str(data) + '\n')
    output_file.flush()


def produce_list(folder_list, fd_train_list, fd_val_train_list):
    total = 0
    off_set = 1
    for i in range(len(folder_list)):

        for num in (list(range(1,31))+list(range(330,360))):
        #for num in (list(range(1,16))+list(range(345,360))):
            color_path = gen_color_path(folder_list[i], num)
            color_lable_path = gen_color_gt_path(folder_list[i], 0)
            depth_path = gen_depth_path(folder_list[i], num)
            depth_lable_path = gen_depth_gt_path(folder_list[i], 0)
            uv_path = gen_uv_path(folder_list[i], num)

            color_path_is_exists = os.path.exists(color_path)
            color_lable_path_is_exists = os.path.exists(color_lable_path)
            depth_path_is_exists = os.path.exists(depth_path)
            depth_lable_path_is_exists = os.path.exists(depth_lable_path)
            uv_path_is_exists = os.path.exists(uv_path)

            if (not color_path_is_exists) and (not color_lable_path_is_exists)\
                    (not depth_path_is_exists) and (not depth_lable_path_is_exists)\
                    and (not uv_path_is_exists):
                break

            if (off_set + num) % TIMES == 0:
                data_str = color_path + ',' + depth_path + ',' + uv_path + ',' + color_lable_path + ',' + depth_lable_path 
                output_data(fd_val_train_list, data_str)
            else:
                data_str = color_path + ',' + depth_path + ',' + uv_path + ',' + color_lable_path + ',' + depth_lable_path 
                output_data(fd_train_list, data_str)

            total = total + 1

    return total

def gen_list(fd_train_list, fd_val_train_list):
    folder_list = ['0000', '0005', '0006', '0014', '0019', '0020', '0025', '0038', '0047', '0049', 
                    '0051','0082', '0083', '0094', '0095',  '0109', '0110', '0112', '0128', '0133', 
                    '0138', '0149', '0161', '0174', '0176', '0180', '0181', '0189', '0192', '0194', 
                    '0200', '0202', '0203', '0206', '0210', '0212', '0218', '0219', '0226', '0230', 
                    '0233', '0243', '0253', '0262', '0266', '0270', '0272', '0274', '0290', '0291', 
                    '0296', '0297', '0298', '0301', '0302', '0303', '0313', '0324', '0328', '0332', 
                    '0335', '0339', '0340', '0345', '0346']
    total = produce_list(folder_list, fd_train_list, fd_val_train_list)
    return total


def main() -> None:
    fd_train_list, fd_val_train_list = open_file()
    total = gen_list(fd_train_list, fd_val_train_list)
    print(total)


if __name__ == '__main__':
    main()
