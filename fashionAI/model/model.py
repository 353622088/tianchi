# coding:utf-8
'''
created on 2018/6/27

@author:sw-git01
'''
import tensorflow as tf
import os
import tensorflow.contrib.slim as slim

from nets import nets_factory

# 可迁移模型
# ['alexnet_v2', 'resnet_v2_101', 'resnet_v2_200', 'mobilenet_v1_075', 'resnet_v1_50', 'nasnet_cifar', 'mobilenet_v1_025',
#  'overfeat', 'vgg_19', 'inception_resnet_v2', 'inception_v3', 'resnet_v2_152', 'resnet_v1_200', 'resnet_v1_152',
#  'mobilenet_v1_050', 'mobilenet_v2', 'nasnet_large', 'inception_v2', 'vgg_a', 'pnasnet_large', 'inception_v4',
#  'cifarnet', 'resnet_v2_50', 'nasnet_mobile', 'resnet_v1_101', 'inception_v1', 'mobilenet_v1', 'lenet', 'vgg_16']
# 有参数模型
# ['inception_v1', 'inception_v2', 'inception_v3', 'inception_v4', 'inception_resnet_v2',
#  'mobilenet_v1_025', 'mobilenet_v1_050', 'mobilenet_v1_100', 'mobilenet_v2_100', 'mobilenet_v2_140',
#  'nasnet_large', 'nasnet_mobile', 'nasnet_plarge',
#  'resnet_v1_50', 'resnet_v1_101', 'resnet_v1_152', 'resnet_v2_50', 'resnet_v2_152',
#  'vgg_16', 'vgg_19']


trained_model_dir = 'D:/model'


class DxqModel(object):
    def __init__(self, net, inputs, inherit=False):
        # inputs = tf.random_uniform((21, 224, 224, 3))
        model_typ = net.split('_')[0]
        pre_trained_model = os.path.join(trained_model_dir, model_typ, net + '.ckpt')
        net_fn = nets_factory.get_network_fn(net)

        self.logits, self.end_points = net_fn(inputs)

        if inherit:
            exclude = nets_factory.excludes_map[net]
            variables_to_restore = slim.get_variables_to_restore(include=None, exclude=exclude)

            # for v in variables_to_restore:
            #     print(v.name.split(":")[0])
            tf.train.init_from_checkpoint(pre_trained_model + '',
                                          {v.name.split(':')[0]: v for v in variables_to_restore})

        self.saver = tf.train.Saver(max_to_keep=30)
