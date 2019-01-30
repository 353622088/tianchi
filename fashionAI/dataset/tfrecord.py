# coding:utf-8 
'''
created on 2019/1/30

@author:Dxq
'''
import os
import tensorflow as tf
import pandas as pd

ALL_CLASSES = ["pant_length", "sleeve_length", "skirt_length", "coat_length",
               "neckline_design", "collar_design", "neck_design", "lapel_design"]
DataRoot = './'


def make_tfrecords(sub_class):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        # tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _write(typ):
        writer = tf.python_io.TFRecordWriter('{}/{}_{}.tfrecords'.format(sub_class, sub_class, typ))
        df = pd.read_csv("{}/{}.csv".format(sub_class, sub_class, 'train'), header=None)
        for i, content in df.iterrows():
            if i == 0: continue

            try:
                img_path = content[0]
                img_path_byte = img_path.encode()
                label = int(content[2].find("y"))
            except:
                print('error')

            # path_byte = str.encode(img_path)
            # image_raw = tf.gfile.FastGFile(img_path, 'rb').read()
            # 等同上面方法

            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(label),
                'image_path': _bytes_feature(img_path_byte),
            }))
            writer.write(example.SerializeToString())

        writer.close()
        print("{}:OK".format(typ))

    for typ in ['train', 'valid']:
        _write(typ)


def step1():
    '''
    :return:按照大类划分数据集
    '''
    df = pd.read_csv("train1_label.csv", header=None)
    df.columns = ['filename', 'label_name', 'label']
    df.label_name = df.label_name.str.replace('_labels', '')

    for sub_class in ALL_CLASSES:
        df_a = df[df['label_name'] == sub_class]
        # df_a = df_a.sample(frac=1).reset_index(drop=True)
        os.makedirs(sub_class, exist_ok=True)
        df_a.to_csv('{}/{}.csv'.format(sub_class, sub_class), index=False, encoding='gbk')


def step2():
    '''
    :return:划分数据集合，每个子类按照百分比(暂时不考虑m的情况)
    '''
    for sub_class in ALL_CLASSES:
        train_df = pd.DataFrame(columns=['filename', 'label_name', 'label'])
        valid_df = pd.DataFrame(columns=['filename', 'label_name', 'label'])
        df = pd.read_csv("{}/{}.csv".format(sub_class, sub_class))
        num_classes = len(df.label[0])
        for i in range(num_classes):
            true_label = 'n' * i + 'y' + 'n' * (num_classes - 1 - i)
            df_b = df[df['label'] == true_label]
            if len(df_b) != 0:
                df_train = df_b.sample(frac=.8, replace=False)
                df_valid = df_b.append(df_train).drop_duplicates(subset=['filename'], keep=False)
                train_df = train_df.append(df_train)
                valid_df = valid_df.append(df_valid)
        os.makedirs(sub_class, exist_ok=True)
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = valid_df.sample(frac=1).reset_index(drop=True)
        train_df.to_csv('{}/{}_train.csv'.format(sub_class, sub_class), index=False, encoding='gbk')
        valid_df.to_csv('{}/{}_valid.csv'.format(sub_class, sub_class), index=False, encoding='gbk')


def step3():
    for sub_class in ALL_CLASSES:
        make_tfrecords(sub_class)


# step1()
# step2()
step3()
