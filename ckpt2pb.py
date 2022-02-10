import glob
import os
import time

import cv2
import tensorflow as tf
import numpy as np


def freeze_graph(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "g_conv_img/BiasAdd,g_conv_mask/BiasAdd"
    # output_node_names = "g_sf/Maximum"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据

        output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点


def prepare_image(img, test_w=-1, test_h=-1):
    if test_w > 0 and test_h > 0:
        img = cv2.resize(np.float32(img), (test_w, test_h), cv2.INTER_CUBIC)
    return img / 255.0


def expand(im):
    if len(im.shape) == 2:
        im = np.expand_dims(im, axis=2)
    im = np.expand_dims(im, axis=0)
    return im


def resize_to_test(img, sz=(640, 480)):
    imw, imh = sz
    return cv2.resize(np.float32(img), (imw, imh), cv2.INTER_CUBIC)


def decode_image(img, resize=False, sz=(640, 480)):
    imw, imh = sz
    img = np.squeeze(np.minimum(np.maximum(img, 0.0), 1.0))
    if resize:
        img = resize_to_test(img, sz=(imw, imh))
    img = np.uint8(img * 255.0)
    if len(img.shape) == 2:
        return np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
    else:
        return img


def pb_test(pb_path, input_img_dir, output_img_dir):
    '''
        :param pb_path:pb文件的路径
        :param image_path:测试图片的路径
        :return:
        '''
    with tf.Graph().as_default():
        output_graph_def = tf.compat.v1.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            for op in sess.graph.get_operations():
                print(op.name, [inp for inp in op.inputs])

            # 定义输入的张量名称,对应网络结构的输入张量
            input_image_tensor = sess.graph.get_tensor_by_name("Placeholder:0")

            # 定义输出的张量名称
            output_tensor_name_img = sess.graph.get_tensor_by_name("g_conv_img/BiasAdd:0")
            output_tensor_name_mask = sess.graph.get_tensor_by_name("g_conv_mask/BiasAdd:0")

            test_h = 400
            test_w = 300
            st = time.time()
            for image_filename in glob.glob(input_img_dir + '/*.jpg'):
                img = cv2.imread(image_filename, -1)
                src = img.copy()

                img = prepare_image(img, test_w, test_h)

                img = expand(img)

                oimg, mask = sess.run([output_tensor_name_img, output_tensor_name_mask],
                                      feed_dict={input_image_tensor: img,
                                                 })
                oimg, mask = decode_image(oimg), decode_image(mask)

                resize_mask = cv2.resize(mask, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_CUBIC)
                resize_oimg = cv2.resize(oimg, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_CUBIC)

                # 形态学处理
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
                resize_mask = cv2.dilate(resize_mask.copy(), kernel) > 0

                src[resize_mask] = resize_oimg[resize_mask]

                if not os.path.isdir(output_img_dir):
                    os.makedirs(output_img_dir)
                output_filename = "%s/%s.png" % (output_img_dir, os.path.splitext(os.path.basename(image_filename))[0])
                cv2.imwrite(output_filename, src)

            print("Test time  = %.3f " % (time.time() - st))


# checkpoint -> .pb
input_checkpoint = 'logs/pre-trained/lasted_model.ckpt'
out_pb_path = "pd_model/frozen_model.pb"
freeze_graph(input_checkpoint, out_pb_path)

# test .pb
# input_img_dir = 'E:/dehw_train_dataset/dehw_testA_dataset/images/'
# output_img_dir = 'E:/dehw_train_dataset/dehw_testA_dataset/masked/'
# pb_test(out_pb_path, input_img_dir, output_img_dir)
