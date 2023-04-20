import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from mlxtend.plotting import plot_confusion_matrix


# 移除额外的标签
def drop_extra_label(df_train, df_test, labels):
    for label in labels:
        df_train.drop(label, axis=1, inplace=True)
        df_test.drop(label, axis=1, inplace=True)

    return pd.concat([df_train, df_test], axis=0)


def encode_string_byte(df, name):
    df[name] = LabelEncoder().fit_transform(df[name])


# 对于离散型特征采用最大最小归一化
def min_max_norm(df, name):
    x = df[name].values.reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df[name] = x_scaled


def numerical_split_ohe(df, name):
    pd_to_np = df[name].tolist()
    np_split = []

    categories = np.linspace(0, 1, num=256, endpoint=False)
    quantization = range(0, 256)

    for value in pd_to_np:
        for i in range(len(categories) - 1):
            if categories[i] <= float(value) <= categories[i + 1]:
                np_split.append(quantization[i])
                break
            if float(value) > categories[-1]:
                np_split.append(quantization[-1])
                break

    df[name] = np_split


# 对数据集进行预处理
def data_preprocess(df):
    # 将proto、state、service、label移到最后几列
    traincols = list(df.columns.values)
    traincols.pop(traincols.index('proto'))
    traincols.pop(traincols.index('state'))
    traincols.pop(traincols.index('service'))
    df = df[traincols + ['proto', 'state', 'service']]

    for i in range(0, len(df.columns.values) - 3):
        min_max_norm(df, df.columns.values[i])

    # 将所有字符型特征进行onehot encoding
    return pd.get_dummies(df, columns=['proto', 'state', 'service'])


def byol_loss(p, z):
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    similarities = tf.reduce_sum(tf.multiply(p, z), axis=1)
    loss = 2 - 2*tf.reduce_mean(similarities)

    return loss, p, z



# def data_preprocess(df, pattern_1, pattern_2):
#     encode_string_byte(df, 'proto')
#     encode_string_byte(df, 'state')
#     encode_string_byte(df, 'service')
#
#     # 将proto、state、service、label移到最后几列
#     traincols = list(df.columns.values)
#     traincols.pop(traincols.index('proto'))
#     traincols.pop(traincols.index('state'))
#     traincols.pop(traincols.index('service'))
#     df = df[traincols + ['proto', 'state', 'service']]
#
#     for i in range(0, len(df.columns.values) - 3):
#         min_max_norm(df, df.columns.values[i])
#
#     for i in range(0, len(df.columns.values) - 3):
#         numerical_split_ohe(df, df.columns.values[i])
#
#     byte_images = np.pad(df.to_numpy(), ((0, 0), (0, 22)), 'constant')
#     x = []
#     for image in np.array(byte_images):
#         x.append((2*image/255 - 1).reshape(8, 8))
#
#     x = np.array(x)
#     # 若算法为gan_densenet、cnn、gan，则需要reshape为[b, 8, 8, 1]
#     if pattern_2 in ['Origin', 'CNN'] or pattern_1 == 'gan':
#         x = x.reshape((x.shape[0], 8, 8, 1))
#     else:
#         x = x.reshape((-1, 64))
#
#     return x


def model_accuracy_loss(history):
    plt.rcParams["figure.figsize"] = (15, 5)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.grid(True)
    plt.show()

    plt.rcParams["figure.figsize"] = (15, 5)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.grid(True)
    plt.show()


def plot_confusing_matrix(y_true, y_pred, n_categories, outcome_labels):
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(n_categories)))
    plot_confusion_matrix(conf_mat=cm, class_names=outcome_labels, figsize=(10, 10), show_normed=True)
    plt.title('Confusing Matrix')
    plt.ylabel('Target')
    plt.xlabel('Predicted')
    plt.show()


def plot_accuracies(nets, history, names, y_accuracy):
    plt.figure(figsize=(15, 5))
    color = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i in range(nets):
        plt.plot(history[i].history['val_accuracy'], linestyle='--', color=color[i])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(names, loc='upper left')
    axes = plt.gca()
    plt.grid(True)
    axes.set_ylim(y_accuracy)
    plt.show()


def gan_train(generator, discriminator, g_optimizer, d_optimizer, db_iter, epochs, batch_sz, z_dim, training=None):
    g_loss_list = []
    d_loss_list = []

    def draw_loss():
        plt.figure()
        plt.subplot(121)
        plt.plot(d_loss_list, 'b')
        plt.title('discriminator')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.subplot(122)
        plt.plot(g_loss_list, 'r')
        plt.title('generator')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    for epoch in range(epochs):
        d_loss_in_loop = []
        g_loss_in_loop = []
        for batch_x in db_iter:
            batch_z = tf.random.uniform([batch_sz, z_dim], maxval=1., minval=-1.)
            # batch_x = tf.cast(tf.reshape(batch_x, [-1, 14, 14, 1]), dtype=tf.float32)
            with tf.GradientTape() as tape:
                d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, training)

            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

            with tf.GradientTape() as tape:
                g_loss = g_loss_fn(generator, discriminator, batch_z, training)

            grads = tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

            d_loss_in_loop.append(d_loss)
            g_loss_in_loop.append(g_loss)

            if epoch % 10 == 0:
                print("epoch", epoch, "fake_loss", float(g_loss), "real_loss", float(d_loss))

        g_loss_list.append(np.mean(g_loss_in_loop))
        d_loss_list.append(np.mean(d_loss_in_loop))

    draw_loss()

    return generator, discriminator


def cross_entropy(logits, is_fake):
    # Least Squared Error
    criterion = tf.keras.losses.MeanSquaredError()
    if is_fake:
        loss = criterion(logits, tf.zeros_like(logits))
    else:
        loss = criterion(logits, tf.ones_like(logits))

    return tf.reduce_mean(loss)


def d_loss_fn(generator, discriminator, batch_z, batch_x, training):
    # 1. treat real image as 1
    # 2. treat generated image as 0
    fake_image = generator(batch_z, training)
    d_real_logits = discriminator(batch_x, training)
    d_fake_logits = discriminator(fake_image, training)

    d_loss_real = cross_entropy(d_real_logits, False)
    d_loss_fake = cross_entropy(d_fake_logits, True)

    return d_loss_fake + d_loss_real


def g_loss_fn(generator, discriminator, batch_z, training):
    fake_image = generator(batch_z, training)
    d_fake_logits = discriminator(fake_image, training)
    loss = cross_entropy(d_fake_logits, False)

    return loss


def random_flip(image):
    # crop = random.randint(11, 14)
    # image = tf.image.random_crop(image, size=[crop, crop, 1])
    left_right_flip = tf.random.uniform(shape=[])
    if left_right_flip < 0.5:
        image = tf.image.random_flip_left_right(image)
    else:
        image = tf.image.random_flip_up_down(image)

    return image

    # return tf.image.resize(image, size=[14, 14])


def random_shuffle(image):
    image = np.reshape(image, 196)
    np.random.shuffle(image)
    image = np.reshape(image, (14, 14, 1))
    left_right_flip = tf.random.uniform(shape=[])
    if left_right_flip < 0.5:
        return tf.image.random_flip_left_right(image)
    else:
        return tf.image.random_flip_up_down(image)

