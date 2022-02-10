from __future__ import division

import cv2

from deshadower import *
import argparse
import glob
import numpy as np
import os
import sys


def prepare_image(img, test_w=-1, test_h=-1):
    if test_w > 0 and test_h > 0:
        img = cv2.resize(np.float32(img), (test_w, test_h), cv2.INTER_CUBIC)
    return img / 255.0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='Dataset/test_dataset/',
                        help="path to sample images")
    parser.add_argument("--result_dir", default='Dataset/result/',
                        help="path to the result dir")
    parser.add_argument("--model", default='logs/pre-trained', type=str,
                        help="path to folder containing the model")
    parser.add_argument("--vgg_19_path", default='Models/imagenet-vgg-verydeep-19.mat', type=str,
                        help="path to vgg 19 path model")
    parser.add_argument("--use_gpu", default=0, type=int, help="which gpu to use")
    parser.add_argument("--is_hyper", default=1, type=int, help="use hypercolumn or not")

    ARGS = parser.parse_args()
    test_h = 480

    deshadower = Deshadower(ARGS.model, ARGS.vgg_19_path, ARGS.use_gpu, ARGS.is_hyper)

    if not os.path.isdir(ARGS.result_dir):
        os.makedirs(ARGS.result_dir)

    for image_filename in glob.glob(ARGS.input_dir + '/*.jpg'):
        print('process: ' + image_filename)
        img = cv2.imread(image_filename, -1)
        src = img.copy()

        test_w = int(img.shape[1] * test_h / float(img.shape[0]))
        img = prepare_image(img, test_w, test_h)

        oimg, mask = deshadower.run(img)

        resize_mask = cv2.resize(mask, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_CUBIC)
        resize_oimg = cv2.resize(oimg, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_CUBIC)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
        mask = cv2.dilate(resize_mask.copy(), kernel) > 0

        src[mask] = resize_oimg[mask]

        if not os.path.isdir(ARGS.result_dir):
            os.makedirs(ARGS.result_dir)
        output_filename = "%s/%s.png" % (ARGS.result_dir, os.path.splitext(os.path.basename(image_filename))[0])
        cv2.imwrite(output_filename, src)
