# -*- coding: UTF-8 -*-

import os
import shutil
import random

import os.path as osp


def seperate_train_val():

    src_root = "../machinery_behavior_recognition/data/input/"
    dst_root = "./data/excavator_cls/"



    val_num = 200



    for cls_name in ['excavator', 'others']:
        src_dir = osp.join(src_root, cls_name)
        src_img_names = os.listdir(src_dir)
        random.shuffle(src_img_names)
        # print(src_img_names)

        src_train_img_names = src_img_names[:-100]
        src_val_img_names = src_img_names[-100:]

        for img_name in src_train_img_names:
            src_img_dir = osp.join(src_dir, img_name)
            dst_img_dir = osp.join(dst_root, 'train', cls_name, img_name)
            shutil.copy(src_img_dir , dst_img_dir)

            # print(src_img_dir, dst_img_dir)

        for img_name in src_val_img_names:
            src_img_dir = osp.join(src_dir, img_name)
            dst_img_dir = osp.join(dst_root, 'val', cls_name, img_name)
            shutil.copy(src_img_dir , dst_img_dir)

            # print(src_img_dir, dst_img_dir)









if __name__ == '__main__':
    seperate_train_val()




