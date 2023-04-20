import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras import layers, Model, optimizers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, scale
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, SVMSMOTE
from util.utils import model_accuracy_loss, plot_confusing_matrix


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
    # t-SNE
    # is_attack = df['attack_cat'].map(lambda a: 'Normal' if a == 'normal' else 'Attack')
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


def data_preprocess_cic(df):

    # 将infinity替换为Nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.dropna(inplace=True)
    # 将label标签变为二分类问题
    # is_attack = df['Label'].map(lambda a: 0 if a == 'BENIGN' else 1)
    # df['Label'] = is_attack

    for i in range(0, len(df.columns.values)-1):
        z_score_norm(df, df.columns.values[i])

    return df


def focal_loss(alpha=0.5, gamma=1.5, epsilon=1e-6):
    def focal_loss_calc(y_true, y_pred):
        positive_pt = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        negative_pt = tf.where(tf.equal(y_true, 0), 1-y_pred, tf.ones_like(y_pred))

        loss = -alpha*tf.pow(1-positive_pt, gamma)*tf.math.log(tf.clip_by_value(positive_pt, epsilon, 1.)) - \
               (1-alpha)*tf.pow(1-negative_pt, gamma)*tf.math.log(tf.clip_by_value(negative_pt, epsilon, 1.))

        return tf.reduce_mean(loss)

    return focal_loss_calc


# df_train = pd.read_csv('../kdd/KDDTrain+.txt')
# # 给数据集每一列添加列名
# df_train = add_labels(df_train)
# # 除去最后一列以及将数据集按照class分为normal(0)和anomaly(1)类别
# df_train = preprocess_labels(df_train)
# # 数据预处理, 最终数据被处理为每条长度为40维
# df_train = data_preprocess(df_train)
#
# df_test = pd.read_csv('../kdd/KDDTest+.txt')
# df_test = add_labels(df_test)
# df_test = preprocess_labels(df_test)
# df_test = data_preprocess(df_test)

csv_path = r'E:\PyCharm\projects\DeepLearning_torch\my_project\datasets\CIC_IDS2017.csv'
df = pd.read_csv(csv_path)
df = data_preprocess_cic(df)
df_Y = pd.get_dummies(df.pop('Label')).values
df_X = df.values.astype(np.float32)

# st = RandomOverSampler(random_state=666)
# st = ADASYN(random_state=666)
st = SMOTE(random_state=666, sampling_strategy={5: 500})
X, y = st.fit_resample(df_X, df_Y)
X = np.reshape(X, (X.shape[0], -1, 1))
print(X.shape, y.shape)


class CNN_LSTM_Fusion1(Model):
    def __init__(self):
        super(CNN_LSTM_Fusion1, self).__init__()
        self.in_shape = layers.Input(shape=(40, 1))
        self.conv1 = layers.Conv1D(16, kernel_size=3, padding='same', activation='tanh')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(32, kernel_size=3, padding='same', activation='tanh')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv1D(64, kernel_size=3, padding='same', activation='tanh')
        self.bn3 = layers.BatchNormalization()
        self.forward_layer = layers.LSTM(70, dropout=0.5, name='lstm')
        self.dense = layers.Dense(6)

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
        self.dense = layers.Dense(1)

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
        self.dense = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        out1 = self.bn1(self.conv1(inputs), training=training)
        out2 = self.bn2(self.conv2(out1), training=training)
        out3 = layers.concatenate([out1, out2])
        out4 = self.bn3(self.conv3(out3), training=training)
        out4 = layers.concatenate([out3, out4])
        out = self.forward_layer(out4)
        out = self.dense(out)

        return tf.nn.sigmoid(out)
#
#
# class CNN_LSTM(Model):
#     def __init__(self):
#         super(CNN_LSTM, self).__init__()
#
#         self.conv1 = layers.Conv1D(16, kernel_size=3, padding='same', activation='tanh')
#         self.bn1 = layers.BatchNormalization()
#         self.conv2 = layers.Conv1D(32, kernel_size=3, padding='same', activation='tanh')
#         self.bn2 = layers.BatchNormalization()
#         self.conv3 = layers.Conv1D(64, kernel_size=3, padding='same', activation='tanh')
#         self.bn3 = layers.BatchNormalization()
#         self.forward_layer = layers.LSTM(70, dropout=0.5)
#         self.dense = layers.Dense(1)
#
#     def call(self, inputs, training=None, mask=None):
#         out1 = self.bn1(self.conv1(inputs), training=training)
#         out2 = self.bn2(self.conv2(out1), training=training)
#         out3 = self.bn3(self.conv3(out2), training=training)
#         out = self.forward_layer(out3)
#         out = self.dense(out)
#         return tf.nn.sigmoid(out)
#
#
# # 消融实验实验的三层CNN模型
class CNN(Model):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = layers.Conv1D(16, kernel_size=3, padding='same', activation='tanh')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(32, kernel_size=3, padding='same', activation='tanh')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv1D(64, kernel_size=3, padding='same', activation='tanh')
        self.bn3 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(6)

    def call(self, inputs, training=None, mask=None):
        out1 = self.bn1(self.conv1(inputs), training=training)
        out2 = self.bn2(self.conv2(out1), training=training)
        out3 = self.bn3(self.conv3(out2), training=training)
        out = self.flatten(out3)
        out = self.dense(out)
        return tf.nn.softmax(out)


# class Lstm(Model):
#     def __init__(self):
#         super(Lstm, self).__init__()
#
#         self.lstm = layers.LSTM(70, dropout=0.5)
#         self.dense = layers.Dense(1)
#
#     def call(self, inputs, training=None, mask=None):
#         out = self.lstm(inputs)
#         out = self.dense(out)
#         return tf.nn.sigmoid(out)
#
#
class Rnn(Model):
    def __init__(self):
        super(Rnn, self).__init__()
        self.rnn = Sequential([
            layers.SimpleRNN(16, dropout=0.5, return_sequences=True),
            layers.SimpleRNN(32, dropout=0.5),
            # layers.SimpleRNN(64, dropout=0.5, return_sequences=True),
            # layers.SimpleRNN(70, dropout=0.5)
        ])
        self.dense = layers.Dense(6)

    def call(self, inputs, training=None, mask=None):

        out = self.rnn(inputs)
        out = self.dense(out)

        return tf.nn.softmax(out)
#
#
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
#         self.conv4 = layers.Conv1D(70, kernel_size=3, padding='same', activation='tanh')
#         self.bn4 = layers.BatchNormalization()
#         self.flatten = layers.Flatten()
#         self.dense = layers.Dense(1)
#
#     def call(self, inputs, training=None, mask=None):
#         out1 = self.bn1(self.conv1(inputs), training=training)
#         out2 = self.bn2(self.conv2(out1), training=training)
#         out3 = self.bn3(self.conv3(out2), training=training)
#         out4 = self.bn4(self.conv4(out3), training=training)
#         out = self.flatten(out4)
#         out = self.dense(out)
#         return tf.nn.sigmoid(out)
#
#
class Fc(Model):
    def __init__(self):
        super(Fc, self).__init__()
        self.rs = layers.Reshape((78, ), input_shape=(78, 1))
        self.dense1 = layers.Dense(70)
        self.dense2 = layers.Dense(64)
        self.dense3 = layers.Dense(32)
        self.dense4 = layers.Dense(16)
        self.ac = layers.Activation('tanh')

        self.dp = layers.Dropout(0.5)
        self.out_layer = layers.Dense(6)

    def call(self, inputs, training=None, mask=None):
        out = self.rs(inputs)
        out = self.ac(self.dense1(out))
        out = self.ac(self.dense2(out))
        out = self.ac(self.dense3(out))
        out = self.ac(self.dense4(out))
        out = self.dp(out)
        out = self.out_layer(out)

        return tf.nn.softmax(out)

# K折交叉验证
# kf = KFold(n_splits=6, shuffle=True, random_state=666)
#
# accuracy_list_in = []
# precision_list_in = []
# recall_list_in = []
# f1_list_in = []

# for train_idx, test_idx in kf.split(X):
#     x_train, y_train = X[train_idx], y[train_idx]
#     x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=666)
#     x_test, y_test = X[test_idx], y[test_idx]
#     print("......")
#     model = CNN_LSTM_Fusion1()
#     optimizer = optimizers.Adam(lr=2e-3)
#     es = EarlyStopping(monitor='accuracy', mode='max', verbose=1, min_delta=0.003, patience=10)
#     model.compile(optimizer=optimizer, loss=focal_loss(), metrics=['accuracy'])
#     history = model.fit(x_train, y_train, validation_data=(x_val, y_val), verbose=1, batch_size=128, epochs=40, callbacks=[es])
#     y_pred = model.predict(x_test)
#     y_pred[y_pred > 0.5] = 1
#     y_pred[y_pred <= 0.5] = 0
#     accuracy_list_in.append(accuracy_score(y_test, y_pred))
#     precision_list_in.append(sum(precision_score(y_test, y_pred, average=None))/2)
#     recall_list_in.append(sum(recall_score(y_test, y_pred, average=None))/2)
#     f1_list_in.append(sum(f1_score(y_test, y_pred, average=None))/2)
#
# acc = sum(accuracy_list_in) / len(accuracy_list_in)
# precision = sum(precision_list_in) / len(precision_list_in)
# recall = sum(recall_list_in) / len(recall_list_in)
# f1 = sum(f1_list_in) / len(f1_list_in)
# print("model avg accuracy is", acc)
# print("model avg precision is", precision)
# print("model avg recall is", recall)
# print("model avg f1 score is", f1)


x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=666)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=666)
model = Rnn()
optimizer = optimizers.Adam(lr=2e-3)
es = EarlyStopping(monitor='accuracy', mode='max', verbose=1, min_delta=0.003, patience=10)
model.compile(optimizer=optimizer, loss=focal_loss(), metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), verbose=1, batch_size=512, epochs=3, callbacks=[es])
y_pred = model.predict(x_test)
# y_pred[y_pred > 0.5] = 1
# y_pred[y_pred <= 0.5] = 0
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
print("accuracy is ", accuracy_score(y_true, y_pred))
print("precision is ", precision_score(y_true, y_pred, average='weighted'))
print("recall is ", recall_score(y_true, y_pred, average='weighted'))
print("f1_score is ", f1_score(y_true, y_pred, average='weighted'))


plot_confusing_matrix(y_true, y_pred, 6, outcome_labels=['Benign', 'GoldenEye', 'Hulk', 'Slowloris', 'Slowhttptest', 'Heartbleed'])


