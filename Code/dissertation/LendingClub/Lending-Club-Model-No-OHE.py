
import pandas as pd
import time
import random

from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from ray.tune.function_runner import StatusReporter
import tensorflow_addons as tfa
from tensorflow.python.keras.optimizers import Adam, sgd

data_info = pd.read_csv('lending_club_info.csv', index_col='LoanStatNew')

print(data_info.loc['revol_util']['Description'])

def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])

feat_info('mort_acc')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('lending_club_loan_two.csv')
df.info()

sns.countplot(x='loan_status',data=df)
plt.figure(figsize=(12,4))
sns.distplot(df['loan_amnt'],kde=False,bins=40)
plt.xlim(0,45000)


plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),annot=True,cmap='viridis')
plt.ylim(10, 0)

feat_info('installment')
feat_info('loan_amnt')
sns.scatterplot(x='installment',y='loan_amnt',data=df,)

sns.boxplot(x='loan_status',y='loan_amnt',data=df)


df.groupby('loan_status')['loan_amnt'].describe()

sorted(df['grade'].unique())
sorted(df['sub_grade'].unique())

sns.countplot(x='grade',data=df,hue='loan_status')

plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' )

plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' ,hue='loan_status')


f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]

plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue='loan_status')


df['loan_status'].unique()

df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})

df[['loan_repaid','loan_status']]


df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
df.head()

len(df)

df.isnull().sum()

# **TASK: Convert this Series to be in term of percentage of the total DataFrame**
100* df.isnull().sum()/len(df)

# **TASK: Let's examine emp_title and emp_length to see whether it will be okay to drop them. Print out their feature information using the feat_info() function from the top of this notebook.**

feat_info('emp_title')
print('\n')
feat_info('emp_length')


# **TASK: How many unique employment job titles are there?**

df['emp_title'].nunique()

df['emp_title'].value_counts()


# **TASK: Realistically there are too many unique job titles to try to convert this to a dummy variable feature. Let's remove that emp_title column.**

df = df.drop('emp_title',axis=1)


# **TASK: Create a count plot of the emp_length feature column. Challenge: Sort the order of the values.**

sorted(df['emp_length'].dropna().unique())

emp_length_order = [ '< 1 year',
                      '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']

plt.figure(figsize=(12,4))

sns.countplot(x='emp_length',data=df,order=emp_length_order)

# **TASK: Plot out the countplot with a hue separating Fully Paid vs Charged Off**
# **CHALLENGE TASK: This still doesn't really inform us if there is a strong relationship between employment length and being charged off, what we want is the percentage of charge offs per category. Essentially informing us what percent of people per employment category didn't pay back their loan. There are a multitude of ways to create this Series. Once you've created it, see if visualize it with a [bar plot](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.plot.html). This may be tricky, refer to solutions if you get stuck on creating this Series.**

emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']

emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']


emp_len = emp_co/emp_fp

emp_len
emp_len.plot(kind='bar')

# **TASK: Charge off rates are extremely similar across all employment lengths. Go ahead and drop the emp_length column.**
df = df.drop('emp_length',axis=1)

# **TASK: Revisit the DataFrame to see what feature columns still have missing data.**

df.isnull().sum()


# **TASK: Review the title column vs the purpose column. Is this repeated information?**
df['purpose'].head(10)
df['title'].head(10)

# **TASK: The title column is simply a string subcategory/description of the purpose column. Go ahead and drop the title column.**
df = df.drop('title',axis=1)
# **NOTE: This is one of the hardest parts of the project! Refer to the solutions video if you need guidance, feel free to fill or drop the missing values of the mort_acc however you see fit! Here we're going with a very specific approach.**

# Find out what the mort_acc feature represents**
feat_info('mort_acc')


# **TASK: Create a value_counts of the mort_acc column.**
df['mort_acc'].value_counts()


# **TASK: There are many ways we could deal with this missing data. We could attempt to build a simple model to fill it in, such as a linear model, we could just fill it in based on the mean of the other columns, or you could even bin the columns into categories and then set NaN as its own category. There is no 100% correct approach! Let's review the other columsn to see which most highly correlates to mort_acc**
print("Correlation with the mort_acc column")
df.corr()['mort_acc'].sort_values()

# **TASK: Looks like the total_acc feature correlates with the mort_acc , this makes sense! Let's try this fillna() approach. We will group the dataframe by the total_acc and calculate the mean value for the mort_acc per total_acc entry. To get the result below:**
print("Mean of mort_acc column per total_acc")
df.groupby('total_acc').mean()['mort_acc']


# **CHALLENGE TASK: Let's fill in the missing mort_acc values based on their total_acc value. If the mort_acc is missing, then we will fill in that missing value with the mean value corresponding to its total_acc value from the Series we created above. This involves using an .apply() method with two columns. Check out the link below for more info, or review the solutions video/notebook.**
# 
# [Helpful Link](https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe) 

# In[ ]:


# CODE HERE


# In[ ]:


total_acc_avg = df.groupby('total_acc').mean()['mort_acc']


# In[ ]:


total_acc_avg[2.0]


# In[ ]:


def fill_mort_acc(total_acc,mort_acc):
    '''
    Accepts the total_acc and mort_acc values for the row.
    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.
    
    total_acc_avg here should be a Series or dictionary containing the mapping of the
    groupby averages of mort_acc per total_acc values.
    '''
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
df.isnull().sum()

# **TASK: revol_util and the pub_rec_bankruptcies have missing data points, but they account for less than 0.5% of the total data. Go ahead and remove the rows that are missing those values in those columns with dropna().**
df = df.dropna()
df.isnull().sum()


# ## Categorical Variables and Dummy Variables
# **We're done working with the missing data! Now we just need to deal with the string values due to the categorical columns.**
# 
# **TASK: List all the columns that are currently non-numeric. [Helpful Link](https://stackoverflow.com/questions/22470690/get-list-of-pandas-dataframe-columns-based-on-data-type)**
# 
# [Another very useful method call](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html)

df.select_dtypes(['object']).columns

# **Let's now go through all the string features to see what we should do with them.**
#
# ### term feature
# 
# **TASK: Convert the term feature into either a 36 or 60 integer numeric data type using .apply() or .map().**
df['term'].value_counts()
df['term'] = df['term'].apply(lambda term: int(term[:3]))

# **TASK: We already know grade is part of sub_grade, so just drop the grade feature.**
df = df.drop('grade',axis=1)

# **TASK: Convert the subgrade into dummy variables. Then concatenate these new columns to the original dataframe. Remember to drop the original subgrade column and to add drop_first=True to your get_dummies call.**

# subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)
# df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)
df['sub_grade'] = df['sub_grade'].apply(lambda grade: ["A", "B", "C", "D", "E", "F", "G"].index(grade[:1]) * 5 + int(grade[1:]))
df.columns

df.select_dtypes(['object']).columns

# ### verification_status, application_type,initial_list_status,purpose 
# **TASK: Convert these columns: ['verification_status', 'application_type','initial_list_status','purpose'] into dummy variables and concatenate them with the original dataframe. Remember to set drop_first=True and to drop the original columns.**

df['verification_status']= df['verification_status'].apply(lambda verified: 0 if verified == "Not Verified" else 1)
df['application_type']= df['application_type'].apply(lambda application: 1 if application == "JOINT" else 0)
df['initial_list_status']= df['initial_list_status'].apply(lambda status: 1 if status == "f" else 0)

dummies = pd.get_dummies(df[['purpose']],drop_first=True)
df = df.drop(['purpose'],axis=1)
# df = pd.concat([df,dummies],axis=1)

df['home_ownership'].value_counts()

df['home_ownership']= df['home_ownership'].apply(lambda ownership: 1 if ownership == "OWN" or ownership == "MORTGAGE" else 0)

# ### address
# **TASK: Let's feature engineer a zip code column from the address in the data set. Create a column called 'zip_code' that extracts the zip code from the address column.**

df['zip_code'] = df['address'].apply(lambda address:address[-5:])
# dummies = pd.get_dummies(df['zip_code'],drop_first=True)
# df = df.drop(['zip_code','address'],axis=1)
# df = pd.concat([df,dummies],axis=1)
df = df.drop(['zip_code','address'],axis=1)

# **TASK: This would be data leakage, we wouldn't know beforehand whether or not a loan would be issued when using our model, so in theory we wouldn't have an issue_date, drop this feature.**
df = df.drop('issue_d',axis=1)

# ### earliest_cr_line
# **TASK: This appears to be a historical time stamp feature. Extract the year from this feature using a .apply function, then convert it to a numeric feature. Set this new data to a feature column called 'earliest_cr_year'.Then drop the earliest_cr_line feature.**

df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
df = df.drop('earliest_cr_line',axis=1)

df.select_dtypes(['object']).columns

# ## Train Test Split
from sklearn.model_selection import train_test_split

df = df.drop('loan_status',axis=1)

# **TASK: Set X and y variables to the .values of the features and label.**

print(df.corr()["loan_repaid"].abs().sort_values(ascending=False)[:30])

X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values
print(len(df))

X_train_unscaled, X_test_unscaled, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_unscaled)
X_test = scaler.transform(X_test_unscaled)

# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm
import kr_helper_funcs as kr
from tensorflow.keras.callbacks import EarlyStopping

rus = RandomUnderSampler()
X_train_rus, y_train_rus = rus.fit_sample(X_train, y_train)

model = Sequential()
model.add(Dense(512,  activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256,  activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128,  activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='sigmoid'))

# optimizer =tfa.optimizers.RectifiedAdam(lr=1e-3)
radam = tfa.optimizers.RectifiedAdam()
ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

model.compile(loss='binary_crossentropy', optimizer= ranger, metrics =['accuracy', kr.f1])

class_weight= {0:1, 1:1}
with_weigths=True
if (with_weigths):
    neg, pos = np.bincount(y_train)
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))
    weight_for_0 = (1 / neg)*(total)/2.0
    weight_for_1 = (1 / pos)*(total)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

early_stop = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(x=X_train,
          y=y_train,
          epochs=30,
          class_weight=class_weight,
          batch_size=256,
          validation_data=(X_test, y_test),
          callbacks=[EarlyStopping(monitor="val_f1", mode='max', patience=5, restore_best_weights=True)]
          )

results = model.evaluate(X_test, y_test)
print(results)

import kr_helper_funcs as kr
from sklearn.metrics import classification_report,confusion_matrix
kr.show_plots(history.history)

predictions = model.predict_classes(X_test)
print(classification_report(y_test, predictions))
kr.plot_cm(y_test, predictions, ["unpaid", "paid"])
plt.show()

# from threading import Lock
# lock = Lock()

import os
def save_model(name = 'without_postcode', model = model):
    # lock.aquire()
    # try:
    import tensorflow as tf
    if not os.path.exists(name):
        os.mkdir(name)
    tf.keras.models.save_model(model, '{}/lending-club.h5'.format(name))
    pd.DataFrame.from_dict(history.history).to_csv(name+"/"+'lending-club-history.csv'.format(name), index=False)
    np.save("{}/X_train_unscaled.npy".format(name), X_train_unscaled)
    np.save("{}/X_test_unscaled.npy".format(name), X_test_unscaled)
    np.save("{}/y_train.npy".format(name), y_train)
    np.save("{}/y_test.npy".format(name), y_test)
    df.to_csv(name+"/"+'lending-club-df.csv'.format(name), index=False)
    # finally:
    #     lock.release()
save_model()




# x, y, hue = "confidence_diff", "class_change", "no_features"
# prop_df = (res[x]
#            .groupby(res[hue])
#            .value_counts(normalize=True)
#            .rename(y)
#            .reset_index())
# plt.figure(figsize=(10,4))
# sns.barplot(x=x, y=y, hue=hue, data=prop_df)#, ax=axes[1])



# **TASK: Create predictions from the X_test set and display a classification report and confusion matrix for the X_test set.**

# import random
# random.seed(101)
# random_ind = random.randint(0,len(df))
# new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
# new_customer
# model.predict_classes(new_customer.values.reshape(1,X_test.shape[1]))
# df.iloc[random_ind]['loan_repaid']


# TUNE-----------------------------------------------------------------------------
from ray.tune import track, tune

import tensorflow.keras as keras
from ray.tune import track
import ray
from ray import tune
class TuneReporterCallback(keras.callbacks.Callback):
    """Tune Callback for Keras.

    The callback is invoked every epoch.
    """

    def __init__(self, logs={}):
        self.iteration = 0
        super(TuneReporterCallback, self).__init__()

    def on_epoch_end(self, batch, logs={}):
        self.iteration += 1
        track.log(keras_info=logs, mean_accuracy=logs.get("accuracy"), mean_loss=logs.get("loss"))


prev_eval =0

rus = RandomUnderSampler()
X_train_rus, y_train_rus = rus.fit_sample(X_train, y_train)

smte = SMOTEENN('auto')
# X_train_smte, y_train_smte = rus.fit_sample(X_train, y_train)
X_train_smte, y_train_smte = smte.fit_sample(X_train, y_train)

from threading import Lock

lock = Lock()


def create_model_tune_cofig(config):
    if config["data"] == "rus":
        X_t = X_train_rus
        y_t = y_train_rus
    elif config["data"] == "smte":
        X_t = X_train_smte
        y_t = y_train_smte
    else:
        X_t = X_train
        y_t = y_train

    create_model_tune(dense_1=int(config["dense_1"]), dense_2=int(config["dense_2"]),
                      batch_size=int(config["batch_size"]),
                      dropout=config["dropout"], optimizer_name=config["optimizer_name"], learning_rate=config["lr"],
                      X_train=X_t, y_train=y_t)

def create_model_tune(dense_1, dense_2, batch_size, dropout, optimizer_name, learning_rate, X_train=X_train, y_train=y_train):
    print ("Params : dense_1:{} dense_2:{} batch_size:{} dropout:{} optimizer_name:{} learning_rate:{}".format(dense_1, dense_2, batch_size, dropout, optimizer_name, learning_rate))
    import tensorflow as tf

    model = tf.keras.Sequential()
    model.add(Dense(dense_1, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(dense_2, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(78, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(39, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(19, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units=1, activation='sigmoid'))

    if optimizer_name == 'ranger':
        radam = tfa.optimizers.RectifiedAdam(learning_rate)
        optimizer = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    elif optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate)
    else:
        optimizer = 'sgd'
    # optimizer = DemonRanger(params=model.parameters())
    model.compile(loss=tf.keras.metrics.binary_crossentropy, optimizer=optimizer, metrics=[tf.keras.metrics.binary_accuracy,  kr.f1])

    # class_weight = {0: 1, 1: 1}
    neg, pos = np.bincount(y_train)
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))
    weight_for_0 = (1 / neg) * (total) / 2.0
    weight_for_1 = (1 / pos) * (total) / 2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    # early_stop = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(x=X_train,
              y=y_train,
              epochs=30,
              class_weight=class_weight,
              batch_size=batch_size,
              validation_data=(X_test, y_test),
              callbacks=[EarlyStopping(monitor="val_f1", mode='max', patience=5, restore_best_weights=True),
                         TuneReporterCallback()]
              )

hyperparameter_space = {
    "dense_1": tune.sample_from(list([256, 512])),
    "dense_2": tune.sample_from(list([90, 128])), # there are 3 other hardcoded layers after this one
    "batch_size": tune.sample_from(list([64, 256, 512, 1024])),
    "dropout": tune.sample_from(list([0.2, 0.5])),
    "optimizer_name": tune.sample_from(list(["adam", "ranger", "sgd"])),
    "lr": tune.sample_from(list([0.001, 0.1])),
    "data": tune.sample_from(list(["default", "rus", "smte"]))
}

num_samples = 4
ray.shutdown()  # Restart Ray defensively in case the ray connection is lost.
ray.init(num_cpus=8, memory=10000000000, log_to_driver=False)
# We clean out the logs before running for a clean visualization later.
# ! rm -rf ~/ray_results/tune_iris

analysis = tune.run(
    create_model_tune_cofig,
    verbose=1,
    config=hyperparameter_space,
    num_samples=num_samples)

# HPARAM-----------------------------------------------------------------------------
# from tensorboard.plugins.hparams import api as hp
# import os
# from datetime import datetime
#
# METRICS = [
#       tf.keras.metrics.TruePositives(name='tp'),
#       tf.metrics.FalsePositives(name='fp'),
#       tf.metrics.TrueNegatives(name='tn'),
#       tf.metrics.FalseNegatives(name='fn'),
#       tf.metrics.BinaryAccuracy(name='accuracy'),
#       tf.metrics.Precision(name='precision'),
#       tf.metrics.Recall(name='recall'),
#       tf.metrics.AUC(name='auc'),
#         kr.f1
# ]
#
# HP_NUM_UNITS_L1 = hp.HParam('num_units_l1', hp.Discrete([1000, 500, 250]))
# HP_NUM_UNITS_L2 = hp.HParam('num_units_l2', hp.Discrete([500, 250]))
# HP_NUM_UNITS_L3 = hp.HParam('num_units_l3', hp.Discrete([250, 128]))
# HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([256,512]))
# HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.2, 0.5))
# HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
# HP_LOSS = hp.HParam('loss', hp.Discrete(['mse', 'binary_crossentropy']))
# HP_WEIGHTS = hp.HParam('weights', hp.Discrete(['True', 'False']))
#
# base_dir = os.path.join('logs', 'hparam_tuning', datetime.now().strftime("%Y-%m-%d-%H%M"))
# if not os.path.exists(base_dir):
#         os.mkdir(base_dir)
#
# prev_eval =0
# def create_model(hparams, logdir, X_train=X_train, y_train=y_train):
#     model = Sequential()
#     model.add(Dense(hparams[HP_NUM_UNITS_L1], activation='relu'))
#     model.add(Dropout(hparams[HP_DROPOUT]))
#     model.add(Dense(hparams[HP_NUM_UNITS_L2], activation='relu'))
#     model.add(Dropout(hparams[HP_DROPOUT]))
#     model.add(Dense(hparams[HP_NUM_UNITS_L3], activation='relu'))
#     model.add(Dropout(hparams[HP_DROPOUT]))
#     model.add(Dense(78, activation='relu'))
#     model.add(Dropout(hparams[HP_DROPOUT]))
#     model.add(Dense(39, activation='relu'))
#     model.add(Dropout(hparams[HP_DROPOUT]))
#     model.add(Dense(19, activation='relu'))
#     model.add(Dropout(hparams[HP_DROPOUT]))
#     model.add(Dense(units=1, activation='sigmoid'))
#     model.compile(loss=hparams[HP_LOSS], optimizer=hparams[HP_OPTIMIZER], metrics=METRICS)
#
#     class_weight= {0:1, 1:1}
#     if (hparams[HP_WEIGHTS]):
#         neg, pos = np.bincount(y_train)
#         total = neg + pos
#         print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
#             total, pos, 100 * pos / total))
#         weight_for_0 = (1 / neg) * (total) / 2.0
#         weight_for_1 = (1 / pos) * (total) / 2.0
#         class_weight = {0: weight_for_0, 1: weight_for_1}
#         print('Weight for class 0: {:.2f}'.format(weight_for_0))
#         print('Weight for class 1: {:.2f}'.format(weight_for_1))
#
#     # early_stop = EarlyStopping(monitor='val_loss', patience=5)
#     model.fit(x=X_train,
#               y=y_train,
#               epochs=30,
#               class_weight=class_weight,
#               batch_size=hparams[HP_BATCH_SIZE],
#               validation_data=(X_test, y_test),
#               callbacks=[EarlyStopping(monitor="val_f1", mode='max', patience=5, restore_best_weights=True),
#                          tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=False,
#                                                         write_images=False, update_freq='epoch',
#                                                         profile_batch=100000000),hp.KerasCallback(logdir, hparams)]
#               )
#
#     global prev_eval
#     evaluation = model.evaluate(X_test, y_test)
#     if evaluation[9] > prev_eval:
#         save_model(name='best', model=model)
#         prev_eval = evaluation[9]
#     return evaluation
#
#
# def run(run_name, hparams, X_train=X_train, y_train=y_train):
#     log_dir = os.path.join(base_dir, run_name)
#     if not os.path.exists(log_dir):
#         os.mkdir(log_dir)
#     with tf.summary.create_file_writer(log_dir).as_default():
#         hp.hparams(hparams)  # record the values used in this trial
#         results = create_model(hparams, log_dir, X_train, y_train)
#         tf.summary.scalar('loss', results[0], step=1)
#         tf.summary.scalar('tp', results[1], step=1)
#         tf.summary.scalar('fp', results[2], step=1)
#         tf.summary.scalar('tn', results[3], step=1)
#         tf.summary.scalar('fn', results[4], step=1)
#         tf.summary.scalar('accuracy', results[5], step=1)
#         tf.summary.scalar('precision', results[6], step=1)
#         tf.summary.scalar('recall', results[7], step=1)
#         tf.summary.scalar('auc', results[8], step=1)
#         tf.summary.scalar('f1-score', results[9], step=1)
#
#
# def hrun(dataset_name='default', X_train=X_train, y_train=y_train):
#     session_num = 0
#     for num_units_l1 in HP_NUM_UNITS_L1.domain.values:
#         for num_units_l2 in HP_NUM_UNITS_L2.domain.values:
#             for num_units_l3 in HP_NUM_UNITS_L3.domain.values:
#                 for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
#                     for optimizer in HP_OPTIMIZER.domain.values:
#                         for loss in HP_LOSS.domain.values:
#                             for batch_size in HP_BATCH_SIZE.domain.values:
#                                 for weights in HP_WEIGHTS.domain.values:
#                                     hparams = {
#                                         HP_NUM_UNITS_L1: num_units_l1,
#                                         HP_NUM_UNITS_L2: num_units_l2,
#                                         HP_NUM_UNITS_L3: num_units_l3,
#                                         HP_DROPOUT: dropout_rate,
#                                         HP_OPTIMIZER: optimizer,
#                                         HP_LOSS: loss,
#                                         HP_BATCH_SIZE: batch_size,
#                                         HP_WEIGHTS:weights}
#                                     run_name = dataset_name + "-run-%d" % session_num
#                                     print('\n--- ----------Starting trial:', run_name, '--------------------------')
#                                     print({h.name: hparams[h] for h in hparams})
#                                     start_time = time.time()
#                                     run(run_name, hparams, X_train, y_train)
#                                     total_time = time.time() - start_time
#                                     print("Total time: {}".format(total_time))
#                                     session_num += 1
#
# hrun()