import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
from tensorflow import keras
from imblearn.combine import SMOTETomek
import tensorboard as tensorboard
import seaborn as seaborn
from tensorflow.python.client import device_lib

print('tensorflow' + tf.__version__)
print('tensorboard' + tensorboard.__version__)
print('seaborn' + seaborn.__version__)
tf.config.list_physical_devices('GPU')
tf.test.is_built_with_cuda
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.list_physical_devices('GPU')
device_lib.list_local_devices()


def loadDataSet():
    return pd.read_csv("../input/stroke-dataset/stroke.csv")
df = loadDataSet()
df = df.drop('id', axis=1)
df.info()


def fill_smoking_status(smoking_status, work_type, gender, age):
    if not pd.isnull(smoking_status):
        return smoking_status
    if work_type == 'children' or gender == 'Female' or age < 18:
        return 'never smoked'
    return 'smokes'


df = loadDataSet()
df['smoking_status'] = df.apply(
    lambda x: fill_smoking_status(x['smoking_status'], x['work_type'], x['gender'], x['age']), axis=1)
smoking_dummies = pd.get_dummies(df['smoking_status'], drop_first=True)
df = pd.concat([df.drop('smoking_status', axis=1), smoking_dummies], axis=1)

df['gender'] = df['gender'].replace(["Other"], "Male")
gender_dummies = pd.get_dummies(df['gender'], drop_first=True)
df = pd.concat([df.drop('gender', axis=1), gender_dummies], axis=1)

bmi_avg = df.groupby('work_type').mean()['bmi']
df['bmi'] = df.apply(lambda x: bmi_avg[x['work_type']] if np.isnan(x['bmi']) else x['bmi'], axis=1)

work_type_dummies = pd.get_dummies(df['work_type'], drop_first=True)
df = pd.concat([df.drop('work_type', axis=1), work_type_dummies], axis=1)

married_dummies = pd.get_dummies(df['ever_married'], drop_first=True)
df = pd.concat([df.drop('ever_married', axis=1), married_dummies], axis=1)
df = df.rename(columns={'Yes': 'Married'})

residence_dummies = pd.get_dummies(df['Residence_type'], drop_first=True)
df = pd.concat([df.drop('Residence_type', axis=1), residence_dummies], axis=1)

df = df.drop('id', axis=1)

from sklearn.model_selection import train_test_split

X = df.drop('stroke', axis=1).values
y = df['stroke'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(np.bincount(y_train))

from tensorboard.plugins.hparams import api as hp
import os

HP_NUM_UNITS_L1 = hp.HParam('num_units_l1', hp.Discrete([32, 49, 79]))
HP_NUM_UNITS_L2 = hp.HParam('num_units_l2', hp.Discrete([32, 16, 8]))
HP_NUM_UNITS_L3 = hp.HParam('num_units_l3', hp.Discrete([8, 4]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([250, 1000]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.4, 0.5))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
HP_LOSS = hp.HParam('loss', hp.Discrete(['mse', 'binary_crossentropy']))

base_dir = os.path.join('logs', 'hparam_tuning', datetime.now().strftime("%Y-%m-%d-%H%M"))
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
        f1_m
]

def train_test_model(model_name, hparams, logdir, X_train=X_train, y_train=y_train, with_weigths=True):
    model = Sequential()
    model.add(Dense(hparams[HP_NUM_UNITS_L1], activation='relu'))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Dense(hparams[HP_NUM_UNITS_L2], activation='relu'))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Dense(hparams[HP_NUM_UNITS_L3], activation='relu'))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss=hparams[HP_LOSS], optimizer=hparams[HP_OPTIMIZER], metrics=METRICS)

    class_weight = {0: 1, 1: 1}
    if with_weigths:
        neg, pos = np.bincount(y_train)
        total = neg + pos
        print('Examples: Total: {} Positive: {} ({:.2f}% of total)'.format(
            total, pos, 100 * pos / total))
        weight_for_0 = (1 / neg) * (total) / 2.0
        weight_for_1 = (1 / pos) * (total) / 2.0
        class_weight = {0: weight_for_0, 1: weight_for_1}
        print('Weight for class 0: {:.2f}'.format(weight_for_0))
        print('Weight for class 1: {:.2f}'.format(weight_for_1))

    model.fit(x=X_train,
              y=y_train,
              epochs=500,
              class_weight=class_weight,
              batch_size=hparams[HP_BATCH_SIZE],
              validation_data=(X_test, y_test),
              verbose=0,
              callbacks=[EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10),
                         tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=False,
                                                        write_images=False, update_freq='epoch',
                                                        profile_batch=100000000),  # log metrics
                         hp.KerasCallback(logdir, hparams)]
              )

    results = model.evaluate(X_test, y_test)
    print(results)
    return results


def run(run_name, hparams, X_train=X_train, y_train=y_train):
    log_dir = os.path.join(base_dir, run_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        results = train_test_model(run_name, hparams, log_dir, X_train, y_train)
        tf.summary.scalar('loss', results[0], step=1)
        tf.summary.scalar('tp', results[1], step=1)
        tf.summary.scalar('fp', results[2], step=1)
        tf.summary.scalar('tn', results[3], step=1)
        tf.summary.scalar('fn', results[4], step=1)
        tf.summary.scalar('accuracy', results[5], step=1)
        tf.summary.scalar('precision', results[6], step=1)
        tf.summary.scalar('recall', results[7], step=1)
        tf.summary.scalar('auc', results[8], step=1)
        tf.summary.scalar('f1-score', results[9], step=1)


def hrun(dataset_name='default', X_train=X_train, y_train=y_train):
    session_num = 0
    for num_units_l1 in HP_NUM_UNITS_L1.domain.values:
        for num_units_l2 in HP_NUM_UNITS_L2.domain.values:
            for num_units_l3 in HP_NUM_UNITS_L3.domain.values:
                for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
                    for optimizer in HP_OPTIMIZER.domain.values:
                        for loss in HP_LOSS.domain.values:
                            for batch_size in HP_BATCH_SIZE.domain.values:
                                hparams = {
                                    HP_NUM_UNITS_L1: num_units_l1,
                                    HP_NUM_UNITS_L2: num_units_l2,
                                    HP_NUM_UNITS_L3: num_units_l3,
                                    HP_DROPOUT: dropout_rate,
                                    HP_OPTIMIZER: optimizer,
                                    HP_LOSS: loss,
                                    HP_BATCH_SIZE: batch_size}
                                run_name = dataset_name + "-run-%d" % session_num
                                start = datetime.now()
                                print('\n--- ----------Starting trial:', run_name, '--------------------------', start)
                                print({h.name: hparams[h] for h in hparams})
                                run(run_name, hparams, X_train, y_train)
                                session_num += 1
                                print('time taken to complete: ', datetime.now() - start)

smt = SMOTETomek('auto')
X_train_SMTomek, y_train_SMTomek = smt.fit_sample(X_train, y_train)
print(np.bincount(y_train_SMTomek))

hrun('SMOTETomek', X_train_SMTomek, y_train_SMTomek)
