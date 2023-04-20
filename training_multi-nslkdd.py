import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import metrics
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from tensorflow.keras import layers, Model, optimizers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, scale
from sklearn.model_selection import train_test_split, KFold
# from util.utils import model_accuracy_loss, plot_confusing_matrix



"""
Introduction to NSL-KDD see https://towardsdatascience.com/a-deeper-dive-into-the-nsl-kdd-data-set-15c753364657
1. Categorical (Features: 2, 3, 4, 42)
2. Binary (Features: 7, 12, 14, 20, 21, 22)
3. Discrete (Features: 8, 9, 15, 23–41, 43)
4. Continuous (Features: 1, 5, 6, 10, 11, 13, 16, 17, 18, 19)
"""


# 在KDDTrain+.arff文件中data没有level这一标签，在KDDTrain+.txt中存在level标签
def add_labels(df):
    df.columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
                  "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
                  "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files",
                  "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                  "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
                  "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
                  "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
                  "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
                  "attack_cat", "level"]

    return df


# 处理二分类问题
def preprocess_labels(df):
    df.drop("level", axis=1, inplace=True)
    is_attack = df['attack_cat'].map(lambda a: 0 if a == 'normal' else 1)
    df['attack_cat'] = is_attack

    return df


# 将字符型特征为数值化
def stoi(df, name):
    df[name] = LabelEncoder().fit_transform(df[name])


# 对于离散型特征采用最大最小归一化
def min_max_norm(df, name):
    x = df[name].values.reshape(-1, 1)
    min_max_scaler = MinMaxScaler()
    x_scaled =min_max_scaler.fit_transform(x)
    df[name] = x_scaled


# 对于连续型特征采用Z-Score方式归一化
def z_score_norm(df, name):
    df[name] = scale(df[name].to_list())


def data_preprocess(df):
    # 将所有字符型特征为数值化
    stoi(df, 'protocol_type')
    stoi(df, 'service')
    stoi(df, 'flag')

    # 将所有连续型特征归一化
    continuous_features_ids = [1, 5, 6, 10, 11, 13, 16, 17, 18, 19]
    for feature_id in continuous_features_ids:
        z_score_norm(df, df.columns[feature_id - 1])

    # 将所有离散型特征归一化
    discrete_features_ids = [8, 9, 15, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
    for feature_id in discrete_features_ids:
        min_max_norm(df, df.columns[feature_id - 1])

    # 由于训练和测试数据集中num_outbound_cmds这一列所有值均为0，故删除此列
    df.drop('num_outbound_cmds', axis=1, inplace=True)

    return df


dos_attacks = ["snmpgetattack", "back", "land", "neptune", "smurf", "teardrop", "pod", "apache2", "udpstorm", "processtable", "mailbomb"]
r2l_attacks = ["snmpguess", "worm", "httptunnel", "named", "xlock", "xsnoop", "sendmail", "ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster"]
u2r_attacks = ["sqlattack", "buffer_overflow", "loadmodule", "perl", "rootkit", "xterm", "ps"]
probe_attacks = ["ipsweep", "nmap", "portsweep", "satan", "saint", "mscan"]

# Normal:[0 1 0 0 0]
# Dos:[1 0 0 0 0]
# Probe:[0 0 1 0 0]
# R2L:[0 0 0 1 0]
# U2R:[0 0 0 0 1]
classes = ["Normal", "Dos", "R2L", "U2R", "Probe"]
label2class = {0: 'Dos', 1: 'Normal', 2: 'Probe', 3: 'R2L', 4: 'U2R'}


def label_attack(row):
    if row["attack_cat"] in dos_attacks:
        return classes[1]
    if row["attack_cat"] in r2l_attacks:
        return classes[2]
    if row["attack_cat"] in u2r_attacks:
        return classes[3]
    if row["attack_cat"] in probe_attacks:
        return classes[4]

    return classes[0]


def focal_loss(alpha=0.5, gamma=1.5, epsilon=1e-6):
    def focal_loss_calc(y_true, y_pred):
        positive_pt = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        negative_pt = tf.where(tf.equal(y_true, 0), 1-y_pred, tf.ones_like(y_pred))

        loss = -alpha*tf.pow(1-positive_pt, gamma)*tf.math.log(tf.clip_by_value(positive_pt, epsilon, 1.)) - \
               (1-alpha)*tf.pow(1-negative_pt, gamma)*tf.math.log(tf.clip_by_value(negative_pt, epsilon, 1.))

        return tf.reduce_mean(loss)

    return focal_loss_calc


df_train = pd.read_csv('../kdd/KDDTrain+.txt')
# 给数据集每一列添加列名
df_train = add_labels(df_train)

df_test = pd.read_csv('../kdd/KDDTest+.txt')
df_test = add_labels(df_test)

# 合并df_train与df_test
df = pd.concat([df_train, df_test], axis=0)
df = data_preprocess(df)
df['attack_cat'] = df.apply(label_attack, axis=1)
df.drop('level', axis=1, inplace=True)
y = pd.get_dummies(df.pop('attack_cat')).values

X = df.values
# st = RandomOverSampler(random_state=666)
st = SMOTE(random_state=666)
X, y = st.fit_resample(X, y)
X = np.reshape(X, (X.shape[0], -1, 1))


class CNN_LSTM_Fusion1(Model):
    def __init__(self):
        super(CNN_LSTM_Fusion1, self).__init__()

        self.conv1 = layers.Conv1D(16, kernel_size=3, padding='same', activation='tanh')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(32, kernel_size=3, padding='same', activation='tanh')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv1D(64, kernel_size=3, padding='same', activation='tanh')
        self.bn3 = layers.BatchNormalization()
        self.forward_layer = layers.LSTM(70, dropout=0.5)
        self.dense = layers.Dense(5)

    def features(self, inputs):
        out1 = self.bn1(self.conv1(inputs))
        out1 = layers.concatenate([inputs, out1])
        out2 = self.bn2(self.conv2(out1))
        out2 = layers.concatenate([inputs, out1, out2])
        out3 = self.bn3(self.conv3(out2))
        out = self.forward_layer(out3)

        return out

    def call(self, inputs, training=None, mask=None):
        out1 = self.bn1(self.conv1(inputs), training=training)
        out1 = layers.concatenate([inputs, out1])
        out2 = self.bn2(self.conv2(out1), training=training)
        out2 = layers.concatenate([inputs, out1, out2])
        out3 = self.bn3(self.conv3(out2), training=training)
        out = self.forward_layer(out3)
        out = self.dense(out)

        return tf.nn.softmax(out)


class CNN_LSTM_Fusion2(Model):
    def __init__(self):
        super(CNN_LSTM_Fusion2, self).__init__()

        self.conv1 = layers.Conv1D(16, kernel_size=3, padding='same', activation='tanh')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(32, kernel_size=3, padding='same', activation='tanh')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv1D(64, kernel_size=3, padding='same', activation='tanh')
        self.bn3 = layers.BatchNormalization()
        self.forward_layer = layers.LSTM(70, dropout=0.5)
        self.dense = layers.Dense(5)

    def call(self, inputs, training=None, mask=None):
        out1 = self.bn1(self.conv1(inputs), training=training)
        out2 = self.bn2(self.conv2(out1), training=training)
        out3 = layers.concatenate([out1, out2])
        out4 = self.bn3(self.conv3(out2), training=training)
        out4 = layers.concatenate([out3, out4])
        out = self.forward_layer(out4)
        out = self.dense(out)

        return tf.nn.sigmoid(out)


class CNN_LSTM_Fusion3(Model):
    def __init__(self):
        super(CNN_LSTM_Fusion3, self).__init__()

        self.conv1 = layers.Conv1D(16, kernel_size=3, padding='same', activation='tanh')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(32, kernel_size=3, padding='same', activation='tanh')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv1D(64, kernel_size=3, padding='same', activation='tanh')
        self.bn3 = layers.BatchNormalization()
        self.forward_layer = layers.LSTM(70, dropout=0.5)
        self.dense = layers.Dense(5)

    def call(self, inputs, training=None, mask=None):
        out1 = self.bn1(self.conv1(inputs), training=training)
        out2 = self.bn2(self.conv2(out1), training=training)
        out3 = layers.concatenate([out1, out2])
        out4 = self.bn3(self.conv3(out3), training=training)
        out4 = layers.concatenate([out3, out4])
        out = self.forward_layer(out4)
        out = self.dense(out)

        return tf.nn.sigmoid(out)


class CNN_LSTM(Model):
    def __init__(self):
        super(CNN_LSTM, self).__init__()

        self.conv1 = layers.Conv1D(16, kernel_size=3, padding='same', activation='tanh')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(32, kernel_size=3, padding='same', activation='tanh')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv1D(64, kernel_size=3, padding='same', activation='tanh')
        self.bn3 = layers.BatchNormalization()
        self.forward_layer = layers.LSTM(70, dropout=0.5)
        self.dense = layers.Dense(5)

    def call(self, inputs, training=None, mask=None):
        out1 = self.bn1(self.conv1(inputs), training=training)
        out2 = self.bn2(self.conv2(out1), training=training)
        out3 = self.bn3(self.conv3(out2), training=training)
        out = self.forward_layer(out3)
        out = self.dense(out)
        return tf.nn.softmax(out)

# 消融实验实验的三层CNN模型
# class CNN(Model):
#     def __init__(self):
#         super(CNN, self).__init__()
#
#         self.conv1 = layers.Conv1D(16, kernel_size=3, padding='same', activation='tanh')
#         self.bn1 = layers.BatchNormalization()
#         self.conv2 = layers.Conv1D(32, kernel_size=3, padding='same', activation='tanh')
#         self.bn2 = layers.BatchNormalization()
#         self.conv3 = layers.Conv1D(64, kernel_size=3, padding='same', activation='tanh')
#         self.bn3 = layers.BatchNormalization()
#         self.flatten = layers.Flatten()
#         self.dense = layers.Dense(5)
#
#     def call(self, inputs, training=None, mask=None):
#         out1 = self.bn1(self.conv1(inputs), training=training)
#         out2 = self.bn2(self.conv2(out1), training=training)
#         out3 = self.bn3(self.conv3(out2), training=training)
#         out = self.flatten(out3)
#         out = self.dense(out)
#         return tf.nn.softmax(out)


# class Lstm(Model):
#     def __init__(self):
#         super(Lstm, self).__init__()
#
#         self.lstm = layers.LSTM(70, dropout=0.5)
#         self.dense = layers.Dense(5)
#
#     def call(self, inputs, training=None, mask=None):
#         out = self.lstm(inputs)
#         out = self.dense(out)
#         return tf.nn.softmax(out)

class Rnn(Model):
    def __init__(self):
        super(Rnn, self).__init__()
        self.rnn = Sequential([
            layers.SimpleRNN(16, dropout=0.5, return_sequences=True),
            layers.SimpleRNN(32, dropout=0.5, return_sequences=True),
            layers.SimpleRNN(64, dropout=0.5, return_sequences=True),
            layers.SimpleRNN(70, dropout=0.5)
        ])
        self.dense = layers.Dense(5)

    def call(self, inputs, training=None, mask=None):
        out = self.rnn(inputs)
        out = self.dense(out)

        return tf.nn.softmax(out)


class CNN(Model):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = layers.Conv1D(16, kernel_size=3, padding='same', activation='tanh')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(32, kernel_size=3, padding='same', activation='tanh')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv1D(64, kernel_size=3, padding='same', activation='tanh')
        self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv1D(70, kernel_size=3, padding='same', activation='tanh')
        self.bn4 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(5)

    def call(self, inputs, training=None, mask=None):
        out1 = self.bn1(self.conv1(inputs), training=training)
        out2 = self.bn2(self.conv2(out1), training=training)
        out3 = self.bn3(self.conv3(out2), training=training)
        out4 = self.bn4(self.conv4(out3), training=training)
        out = self.flatten(out4)
        out = self.dense(out)
        return tf.nn.softmax(out)


class Fc(Model):
    def __init__(self):
        super(Fc, self).__init__()
        self.rs = layers.Reshape((40, ), input_shape=(40, 1))
        self.dense1 = layers.Dense(70)
        self.dense2 = layers.Dense(64)
        self.dense3 = layers.Dense(32)
        self.dense4 = layers.Dense(16)
        self.ac = layers.Activation('tanh')

        self.dp = layers.Dropout(0.5)
        self.out_layer = layers.Dense(5)

    def call(self, inputs, training=None, mask=None):
        out = self.rs(inputs)
        out = self.ac(self.dense1(out))
        out = self.ac(self.dense2(out))
        out = self.ac(self.dense3(out))
        out = self.ac(self.dense4(out))
        out = self.dp(out)
        out = self.out_layer(out)

        return tf.nn.softmax(out)


# t-sne可视化
# x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=666)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=666)
# model = CNN_LSTM_Fusion1()
# optimizer = optimizers.Adam(lr=2e-3)
# es = EarlyStopping(monitor='accuracy', mode='max', verbose=1, min_delta=0.003, patience=10)
# model.compile(optimizer=optimizer, loss=focal_loss(), metrics=['accuracy'])
# history = model.fit(x_train, y_train, validation_data=(x_val, y_val), verbose=1, batch_size=128, epochs=40, callbacks=[es])
#
# random_ids = np.random.randint(0, x_test.shape[0], 10000)
# ts = TSNE(n_components=2, random_state=666)
# data = x_test[random_ids]
# predict_result = model.features(data)
# y_pred = model.predict(data)
# y_pred = np.argmax(y_pred, axis=1)
# res = ts.fit_transform(predict_result)
# markers = ['s', 'd', 'o', '^', 'v']
# color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'cyan'}
# plt.figure(dpi=500)
# fig = plt.figure()
# ax = plt.subplot(111)
# for idx, label in enumerate(np.unique(y_pred)):
#     plt.scatter(res[y_pred == label, 0], res[y_pred == label, 1], c=color_map[idx], marker=markers[idx], label=label2class[label])
#
# plt.xlabel('X in t-SNE')
# plt.ylabel('Y in t-SNE')
# plt.legend(loc='upper left')
# plt.title('t-SNE visualization of NSL-KDD processed by MCST-IDS')
# plt.show()
# from sklearn.manifold import TSNE
#
# random_ids = np.random.randint(0, X.shape[0]+1, 20000)
# ts = TSNE(n_components=2, random_state=666)
# data = X[random_ids]
# labels = y[random_ids]
# res = ts.fit_transform(data)
# markers = ['s', 'd', 'o', '^', 'v']
# color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'cyan'}
# fig = plt.figure()
# ax = plt.subplot(111)
# for idx, label in enumerate(np.unique(labels)):
#     plt.scatter(res[labels == label, 0], res[labels == label, 1], c=color_map[idx], marker=markers[idx], label=label)
#
# plt.xlabel('X in t-SNE')
# plt.ylabel('Y in t-SNE')
# plt.legend(loc='upper left')
# plt.title('t-SNE visualization of NSL-KDD')
# plt.show()


# 使用k-fold进行交叉验证
kf = KFold(n_splits=6, shuffle=True, random_state=666)

accuracy_list_in = []
precision_list_in = []
recall_list_in = []
f1_list_in = []

for train_idx, test_idx in kf.split(X):
    x_train, y_train = X[train_idx], y[train_idx]
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=666)
    x_test, y_test = X[test_idx], y[test_idx]
    model = CNN_LSTM_Fusion1()
    optimizer = optimizers.Adam(lr=2e-3)
    es = EarlyStopping(monitor='accuracy', mode='max', verbose=1, min_delta=0.003, patience=10)
    model.compile(optimizer=optimizer, loss=focal_loss(), metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), verbose=1, batch_size=128, epochs=1, callbacks=[es])
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    # plot_confusing_matrix(y_true, y_pred, 5, classes)
    accuracy_list_in.append(metrics.accuracy_score(y_true, y_pred))
    precision_list_in.append(metrics.precision_score(y_true, y_pred, average=None))
    recall_list_in.append(metrics.recall_score(y_true, y_pred, average=None))
    f1_list_in.append(metrics.f1_score(y_true, y_pred, average=None))

acc = sum(accuracy_list_in) / len(accuracy_list_in)
precision = sum(precision_list_in) / len(precision_list_in)
recall = sum(recall_list_in) / len(recall_list_in)
f1 = sum(f1_list_in) / len(f1_list_in)
print("model avg accuracy is", acc)
print("model avg precision is", precision)
print("model avg recall is", recall)
print("model avg f1 score is", f1)

# 不同K值对多分类指标影响
# def example_plot(ax, y, ylabel, fontsize=12):
#     x = range(2, 11, 2)
#     ax.plot(x, y)
#     ax.scatter(x, y)
#     ax.set_xlabel('K value', fontsize=fontsize)
#     ax.set_ylabel(ylabel, fontsize=fontsize)
#     ax.set_title(ylabel+' on different K values', fontsize=fontsize)
#     ax.grid(True)
#
#
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# example_plot(ax1, accuracy_list, 'Accuracy')
# example_plot(ax2, precision_list, 'Precision')
# example_plot(ax3, recall_list, 'Recall')
# example_plot(ax4, f1_list, 'F1_score')
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# plt.show()














