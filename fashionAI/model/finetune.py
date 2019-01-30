# coding:utf-8
'''
created on 2018/5/22

@author:sw-git01
'''
import os
import sys
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import shutil
from scipy import misc

from model import DxqModel

flags = tf.flags
flags.DEFINE_bool('train', True, 'Either train the model or test the model.')
flags.DEFINE_integer('batch_size', 250, 'Batch size')

flags.DEFINE_integer('train_steps', 1000, 'steps of training')
flags.DEFINE_integer('test_steps', 100, 'steps of testing')
# flags.DEFINE_string('test_data', "test", 'typ of testing')

flags.DEFINE_integer('img_height', 112, 'height size')
flags.DEFINE_integer('img_depth', 3, 'depth size')
flags.DEFINE_integer('num_classes', 8, 'num of classes')

flags.DEFINE_float('min_lr', 5e-6, 'the minimum value of  lr')
flags.DEFINE_float('start_lr', 1e-3, 'the start value of  lr')
flags.DEFINE_float('dropout_rate', .5, 'Dropout rate')

flags.DEFINE_string('data_dir', '../dataset/coat_length/coat_length_train.tfrecords', 'The data directory.')
flags.DEFINE_string('summary_dir', '../summary', 'The summary dir')
FLAGS = flags.FLAGS

_NUM_IMAGES = {
    'train': 8715,
    'valid': 2179
}


def random_rotate_image_func(image):
    # 旋转角度范围
    angle = np.random.uniform(low=-6.0, high=6.0)
    return misc.imrotate(image, angle, 'bicubic')


def parser(record, aug):
    keys_to_features = {
        'image_path': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    image_raw = tf.read_file(FLAGS.data_dir + '/' + parsed['image_path'])
    image = tf.image.decode_jpeg(image_raw)
    image = tf.image.resize_images(image, [FLAGS.img_height, FLAGS.img_height])
    image = tf.reshape(image, [FLAGS.img_height, FLAGS.img_height, FLAGS.img_depth])
    image = tf.cast(image, tf.float32)

    # image = tf.random_crop(image, [124, 124, 3])
    if aug:
        # image = tf.image.random_hue(image, max_delta=0.05)
        # if np.random.random() < 0.5:
        #     image = tf.image.flip_left_right(image)
        if np.random.random() < 0.7:
            image = tf.image.random_contrast(image, lower=.8, upper=1.2)
        if np.random.random() < 0.7:
            image = tf.image.random_brightness(image, max_delta=0.2)
            # image = tf.py_func(random_rotate_image_func, [image], tf.uint8)
        if np.random.random() < 0.5:
            image = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
        # image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

    image = tf.cast(image, tf.float32)
    # image = tf.image.per_image_standardization(image)
    label = tf.cast(parsed['label'], tf.int32)
    return image, label, parsed['image_path']


def get_record_data(graph, typ, aug=True):
    with graph.as_default():
        if typ == 'train':
            dataset = tf.data.TFRecordDataset([os.path.join(FLAGS.data_dir, '{}.tfrecords'.format(typ))])
        else:
            dataset = tf.data.TFRecordDataset(os.path.join(FLAGS.data_dir, '{}.tfrecords'.format(typ)))
        dataset = dataset.map(lambda x: parser(x, aug))

        num_epochs = -1 if typ == 'train' else 1
        # num_epochs = 1
        batch_size = FLAGS.batch_size
        # buffer_size = _NUM_IMAGES[typ]
        # dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size).repeat(num_epochs)
        dataset = dataset.repeat(num_epochs).batch(batch_size)

        iterator = dataset.make_one_shot_iterator()

        features, labels, paths = iterator.get_next()
        # features = (features - 128) / 68.0
        return features, labels, paths


def extract_step(path):
    file_name = os.path.basename(path)
    print(file_name)
    return int(file_name.split('-')[-1])


def loader(saver, session, load_dir):
    print('load_dir', load_dir)
    print(tf.gfile.Exists(load_dir))
    if tf.gfile.Exists(load_dir):
        ckpt = tf.train.get_checkpoint_state(load_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            prev_step = extract_step(ckpt.model_checkpoint_path)

        else:
            tf.gfile.DeleteRecursively(load_dir)
            tf.gfile.MakeDirs(load_dir)
            prev_step = 0
    else:
        tf.gfile.MakeDirs(load_dir)
        prev_step = 0
    return prev_step


def finetune_model(base_out):
    net = tf.layers.flatten(base_out)
    # net = tf.layers.dense(inputs=net, units=2, activation=tf.nn.relu, name='DENSE1')
    # net = tf.layers.dropout(inputs=net, rate=FLAGS.dropout_rate)
    out = tf.layers.dense(inputs=net, units=FLAGS.num_classes, name='OUT')
    print(out)
    return out


def summary(var_list, loss, out, Y, load_dir):
    # Add the variables we train to the summary
    for var in var_list:
        tf.summary.histogram(var.name, var)

    # Add the loss to summary
    tf.summary.scalar('cross_entropy', loss)

    # Evaluation op: Accuracy of the model
    with tf.name_scope("accuracy"):
        predict_op = tf.argmax(out, 1)
        print(predict_op)
        correct_pred = tf.equal(predict_op, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Add the accuracy to the summary
    tf.summary.scalar('accuracy', accuracy)

    # Merge all summaries together
    merge_op = tf.summary.merge_all()

    return merge_op, accuracy


def param_report(graph):
    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        graph, tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)

    sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)


def train():
    graph = tf.Graph()
    with graph.as_default():
        class_num = FLAGS.num_classes
        load_dir = FLAGS.summary_dir + '/train/'

        # features, labels = get_record_data(graph, 'train')
        features = tf.placeholder(tf.float32, shape=[None, FLAGS.img_height, FLAGS.img_height, FLAGS.img_depth],
                                  name='features')
        tf.summary.image('inputs', features)
        print(features)
        labels = tf.placeholder(tf.int32, shape=[None, ], name='labels')
        Y = tf.one_hot(labels, depth=class_num, axis=1, dtype=tf.float32)

        # extend others model
        # model = DxqModel(features)
        model = DxqModel('alexnet_v2', features, inherit=False)
        # 根据模型需要修改
        print(model.end_points.keys())
        net = model.end_points['alexnet_v2/conv5']

        # for key in model.end_points.keys():
        #     print(key)
        # 后续的模型结构
        out = finetune_model(net)
        predict_op = tf.argmax(input=out, axis=1, name='classes')
        print(predict_op)
        loss = tf.losses.softmax_cross_entropy(Y, out, scope='LOSS', weights=1.0)

        var_list = []
        global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        learning_rate = tf.train.exponential_decay(FLAGS.start_lr, global_step=global_step, decay_steps=50,
                                                   decay_rate=0.9)
        learning_rate = tf.maximum(learning_rate, FLAGS.min_lr)
        tf.summary.scalar('learning_rate', learning_rate)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

        # param_report
        # param_report(graph)

        merge_op, accuracy = summary(var_list, loss, out, Y, load_dir)

        # Initialize an saver for store model checkpoints
        saver = tf.train.Saver(max_to_keep=20)

        # Get the number of training/validation steps per epoch
        train_batches_per_epoch = int(np.floor(_NUM_IMAGES['train'] / FLAGS.batch_size))
        val_batches_per_epoch = int(np.floor(_NUM_IMAGES['valid'] / FLAGS.batch_size))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            # Loop over number of epochs

            last_step = loader(saver, sess, load_dir)
            max_steps = FLAGS.train_steps

            # Initialize the FileWriter
            train_writer = tf.summary.FileWriter(load_dir, graph=sess.graph)
            valid_writer = tf.summary.FileWriter(load_dir.replace("train", "valid"), graph=sess.graph)
            for epoch in range(last_step, max_steps):
                tr_features, tr_labels, _ = get_record_data(graph, 'train')
                # tr_features, tr_labels = get_record_data(graph, 'train')
                # te_features, te_labels = get_record_data(graph, 'valid')

                train_acc = 0.
                train_count = 0
                print('epoch:', epoch + 1, '/', max_steps)
                for step in tqdm(range(train_batches_per_epoch)):
                    # get next batch of data
                    tr_feature_batch, tr_label_batch = sess.run([tr_features, tr_labels])
                    # And run the training op
                    a, _, tr_acc = sess.run([loss, train_op, accuracy],
                                            feed_dict={features: tr_feature_batch, labels: tr_label_batch})
                    print(a)
                    # Generate summary with the current batch of data and write to file

                    s = sess.run(merge_op, feed_dict={features: tr_feature_batch, labels: tr_label_batch})
                    train_writer.add_summary(s, epoch * train_batches_per_epoch + step)
                    if step < 39:
                        train_acc += tr_acc * len(tr_label_batch)
                        train_count += len(tr_label_batch)
                    del tr_feature_batch
                    del tr_label_batch
                train_acc /= train_count
                _S1 = tf.Summary(value=[tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])
                train_writer.add_summary(summary=_S1, global_step=epoch + 1)
                print("Train Accuracy = {:.4f}".format(train_acc))

                test_acc = 0.
                test_count = 0
                # if (epoch + 1) % 5 == 0:
                te_features, te_labels, _ = get_record_data(graph, 'valid', False)
                for _ in range(val_batches_per_epoch + 1):
                    try:
                        te_feature_batch, te_label_batch = sess.run([te_features, te_labels])
                        acc = sess.run(accuracy, feed_dict={features: te_feature_batch,
                                                            labels: te_label_batch})
                        s = sess.run(merge_op, feed_dict={features: te_feature_batch, labels: te_label_batch})
                        valid_writer.add_summary(s, epoch * val_batches_per_epoch + _)

                        test_acc += acc * len(te_label_batch)
                        test_count += len(te_label_batch)
                        del te_feature_batch
                        del te_label_batch
                    except tf.errors.OutOfRangeError:
                        print(_)
                test_acc /= test_count
                _S2 = tf.Summary(value=[tf.Summary.Value(tag='valid_accuracy', simple_value=test_acc)])
                valid_writer.add_summary(summary=_S2, global_step=epoch + 1)
                print("Validation Accuracy = {:.4f}".format(test_acc))


                # save checkpoint of the model
                saver.save(sess, os.path.join(load_dir, 'model.ckpt'), global_step=epoch + 1)

            train_writer.close()
            valid_writer.close()


def predict(error_show=False):
    graph = tf.get_default_graph()
    model_epoch = 358
    model_path = FLAGS.summary_dir + '/train/model.ckpt-{}'.format(model_epoch)
    saver = tf.train.import_meta_graph("{}.meta".format(model_path))
    sess = tf.InteractiveSession(graph=graph)
    saver.restore(sess, model_path)
    predict_op = graph.get_tensor_by_name("classes:0")
    features = graph.get_tensor_by_name("features: 0")

    # if (epoch + 1) % 5 == 0:
    test_record_name = 'valid'
    te_features, te_labels, te_paths = get_record_data(graph, test_record_name, False)
    nums = int(np.floor(_NUM_IMAGES[test_record_name] / FLAGS.batch_size))
    res_matrix = np.zeros([3, 3])
    for _ in range(nums + 1):
        try:
            te_feature_batch2, te_label_batch2, te_path2 = sess.run([te_features, te_labels, te_paths])
            prediction = predict_op.eval(feed_dict={features: te_feature_batch2})
            matrix = confusion_matrix(te_label_batch2, prediction, labels=[0, 1, 2])
            res_matrix += matrix
            if error_show:
                for i in range(len(te_label_batch2)):
                    if prediction[i] != te_label_batch2[i]:
                        path = te_path2[i].decode()
                        file_path = os.path.join(FLAGS.data_dir, path)
                        file_name = os.path.basename(file_path)
                        error_dir = os.path.join(FLAGS.data_dir, 'result', 'error-' + str(model_epoch),
                                                 str(te_label_batch2[i]), str(prediction[i]))
                        os.makedirs(error_dir, exist_ok=True)
                        shutil.copy(file_path, os.path.join(error_dir, file_name))
                        # Image.fromarray(np.uint8(te_feature_batch2[i] * 58 + 120), 'RGB').save(
                        #     'error/pre{}-cor{}-{}-{}.jpg'.format(prediction[i], te_label_batch2[i], i, _))
        except tf.errors.OutOfRangeError:
            print(_)

    print(res_matrix)
    print(np.sum(res_matrix))
    print(round((np.sum(res_matrix) - np.trace(res_matrix)) * 100 / np.sum(res_matrix), 3))


def main(_):
    if FLAGS.train:
        train()
    else:
        predict(error_show=True)


if __name__ == '__main__':
    tf.app.run()
