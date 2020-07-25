import tensorflow as tf
import shap
from lime import lime_tabular
import numpy as np
import random
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import kr_helper_funcs as kr
from eli5.sklearn import PermutationImportance
import tensorflow_addons as tfa
import sys
import csv
import os

from collections import defaultdict

# load values specific to saved model using the saved name
name = 'mouse'
fileName = "{}/{}_{}.csv".format(name, sys.argv[1], name)
newrows_filename = "{}/{}_{}_rows.csv".format(name, sys.argv[1], name)
radam = tfa.optimizers.RectifiedAdam()
ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
model = tf.keras.models.load_model('{}/{}.h5'.format(name,name), custom_objects={"f1": kr.f1, "optimizer": ranger})
X_train_unscaled = np.load("{}/X_train_unscaled.npy".format(name))
X_test_unscaled = np.load("{}/X_test_unscaled.npy".format(name))
y_train = np.load("{}/y_test.npy".format(name))
y_test = np.load("{}/y_test.npy".format(name))
df = pd.read_csv('{}/{}-df.csv'.format(name,name))

#scale data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_unscaled)
X_test = scaler.transform(X_test_unscaled)

prev_scaled_row = None
cached_map_values = None
map_values_eli5 = defaultdict(list)

shape_size = X_train.shape[1]

#used for distribution strategy (map indices of values where the label is 0 and 1  or higher
y_test_map = defaultdict(list)
for i in range(max(y_test)+1):
    y_test_map[i] = np.array(np.where(y_test==i))[0]
y_test_map

#used for distribution others - create a set of unique values for each feature specific to each class
X_test_map = defaultdict(list)
# sort rows by the true class
for i in range(len(X_test)):
     X_test_map[y_test[i]].append(X_test[i])
# transpose rows within a class and map the values for each feature to a set
for i in X_test_map.keys():
    X_test_map[i] = [list(set(a)) for a in np.array(X_test_map[i]).transpose()]
X_test_map


def get_correctly_predicted_indices(num_rows):
    predictions = model.predict_classes(X_test)
    # predictions_flat = [item for sublist in predictions for item in sublist]
    equal_indices= predictions ==y_test
    indices = np.array(np.where(equal_indices==True))
    return indices[0]

#return a map of immportant column indices specific to a datapoint
def explain_row_lime(scaled_row, explainer, nsamples=100, verbose=0):
    if nsamples == "auto":
        exp = explainer.explain_instance(scaled_row, model.predict, num_features=30,
                                         top_labels=30)  # labels=list(df.columns)
    else:
        exp = explainer.explain_instance(scaled_row, model.predict, num_features=30, top_labels=30,
                                         num_samples=nsamples)  # labels=list(df.columns)

    if (verbose == 1):
        exp.show_in_notebook(show_table=True, show_all=False)

    return exp.as_map()

#return a map of immportant column indices specific to a datapoint
def explain_row_shap(scaled_row, explainer, nsamples=100, verbose=0):
    shap_values = explainer.shap_values(scaled_row.reshape(1, shape_size), nsamples=nsamples, l1_reg="num_features(30)")

    if (verbose == 1):
        shap.decision_plot(explainer.expected_value[0], shap_values[0][0, :], scaled_row,
                           feature_names=list(df.drop('loan_repaid', axis=1).columns), link="logit")

    map_values = {}
    for class_value in range(len(shap_values)):
        s = shap_values[class_value][0]
        sorted_indices = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
        #         print(sorted_indices)
        ordered_list = [(a, shap_values[class_value][0][a]) for a in sorted_indices if
                        shap_values[class_value][0][a] > 0]
        map_values[class_value] = ordered_list
    #     print(map_values)
    return map_values

#return a map of immportant column indices specific to a datapoint - this is a dummy random explainer
def explain_row_random(scaled_row, explainer, nsamples=100, verbose=0):
    # generate map of random explanations for a row
    # random_list = {}
    random_list = [(i, random.uniform(0.0, 1.0)) for i in range(shape_size)]
    # random_list[1] = [(i, random.uniform(0.0, 1.0)) for i in range(shape_size)]

    map_values = {}
    for class_value in range(max(y_test)):
        s = random_list
        sorted_indices = sorted(range(len(s)), key=lambda k: s[k][1], reverse=True)
        #         print(sorted_indices)
        ordered_list = [(a, random_list[a][1]) for a in sorted_indices if
                        random_list[a][1] > 0]
        map_values[class_value] = ordered_list
    #     print(map_values)
    return map_values

#explain rows using PermutationImportance. This provides a set of global explanation computed once only and returns the
#same values regardless of the datapoint
def explain_row_eli5():
    global map_values_eli5

    # compute explanations only once
    if map_values_eli5 != None:
        return map_values_eli5

    copy_model = tf.keras.models.load_model('{}/{}.h5'.format(name,name), custom_objects={"f1": kr.f1})

    def base_model():
        return copy_model

    my_model = KerasClassifier(build_fn=base_model)
    my_model.fit(X_test.copy(), y_test.copy())

    perm = PermutationImportance(my_model).fit(X_test.copy(), y_test.copy())
    # eli5.show_weights(perm, feature_names=list(df.drop('loan_repaid', axis=1).columns))

    s = perm.feature_importances_
    sorted_indices = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
    class_1 = [(a, s[a]) for a in sorted_indices if s[a] > 0]
    sorted_indices = sorted(range(len(s)), key=lambda k: s[k])
    class_0 = [(a, s[a] * -1) for a in sorted_indices if s[a] <= 0]

    for class_value in range(max(y_test)):
        map_values_eli5[class_value] = class_1

    return map_values_eli5

# strategy to replace the important columns in a row with the mean values: mean (mean of 0 and mean of 1)
def recreate_row_mean(neutral_points, unscaled_row, map_values, predicted_class, no_exp=3, verbose=0, feature_ranking='first'):
    important_columns = extract_important_columns(feature_ranking, map_values, no_exp, predicted_class)

    for c in important_columns:
        unscaled_row[c] = neutral_points[c]

    if (verbose == 1):
        print(pd.DataFrame(unscaled_row))

    return scaler.transform(pd.DataFrame(unscaled_row).values.reshape(1, shape_size)), important_columns

#strategy to replace important columns with a random value from the distribution specific to a class - this will be called several times for each point
def recreate_row_from_distribution_same(scaled_row, which_side, map_values, predicted_class, no_exp=3, verbose=0, feature_ranking='first'):
    important_columns = extract_important_columns(feature_ranking, map_values, no_exp, predicted_class)

    for c in important_columns:
        index = y_test_map[which_side][random.randint(0, y_test_map[which_side].shape[0]-1)]
        scaled_row[c] = X_test[index][c]

    if (verbose == 1):
        print(pd.DataFrame(scaled_row))

    return pd.DataFrame(scaled_row).values.reshape(1, shape_size), important_columns

# this strategy replaces all other columns except the important ones
def recreate_row_from_distribution_other_features(scaled_row, which_side, map_values, predicted_class, no_exp=3, verbose=0, feature_ranking='first'):
    important_columns = extract_important_columns(feature_ranking, map_values, no_exp, predicted_class)

    for i in range(0, len(scaled_row)):
        if (i not in important_columns) :
            scaled_row[i] = X_test_map[which_side][i][random.randint(0, len(X_test_map[which_side][i])-1)]

    if (verbose == 1):
        print(pd.DataFrame(scaled_row))

    return pd.DataFrame(scaled_row).values.reshape(1, shape_size), important_columns


def extract_important_columns(feature_ranking, map_values, no_exp, predicted_class):
    tuple_list = map_values[predicted_class]
    important_columns = []
    if no_exp > 0:
        if feature_ranking == 'first':
            important_columns = [a for (a, b) in tuple_list if b > 0][:no_exp]
        elif feature_ranking == "middle":
            important_columns = [a for (a, b) in tuple_list if b > 0]
            midpoint = int(len(important_columns) / 2)
            start_middle = midpoint - int(no_exp / 2)
            important_columns = important_columns[start_middle:start_middle + no_exp]
        elif feature_ranking == "last":
            important_columns = [a for (a, b) in tuple_list if b > 0][-no_exp:]
    return important_columns


def calculate_probability_diff(row_number, neutral_points, explainer, no_exp=3, verbose=0, which_explainer='lime',
                               nsamples=100, feature_ranking='first', number_of_elimination = 20, strategy="distribution"):
    scaled_row = X_test[row_number].copy()
    # unscaled_row = X_test_unscaled[row_number].copy()

    reshaped_scaled_row = pd.DataFrame(scaled_row).values.reshape(1, shape_size)
    original_class = model.predict_classes(reshaped_scaled_row)[0]
    original_probability = model.predict_proba(reshaped_scaled_row)
    original_predict_fn = model.predict(reshaped_scaled_row)

    if (original_class != y_test[row_number]):
        print("Something went wrong! Original class is different than y_test")
        return

    global prev_scaled_row
    global cached_map_values
    if ((scaled_row == prev_scaled_row).all()):
        # print("Hit cache!")
        map_values = cached_map_values;
    elif (which_explainer == 'lime'):
        map_values = explain_row_lime(scaled_row, explainer, nsamples=nsamples, verbose=verbose)
    elif (which_explainer == 'shap'):
        map_values = explain_row_shap(scaled_row, explainer, nsamples=nsamples, verbose=verbose)
    elif (which_explainer == 'random'):
        map_values = explain_row_random(scaled_row, explainer, nsamples=nsamples, verbose=verbose)
    elif (which_explainer == 'eli5'):
        map_values = explain_row_eli5(); # get global explanations

    prev_scaled_row = scaled_row
    cached_map_values = map_values
    if (verbose == 1):
        print(map_values)

    important_columns=None
    new_class = None
    new_predict_fn = None
    new_probability = None
    new_row = None
    if strategy == 'distribution' or strategy == 'distribution_others':
        save_row(row_number, original_class, scaled_row)
        important_columns, new_class, new_predict_fn, new_probability, new_row = recreate_row_and_get_new_probabilities_ditribution(
            feature_ranking, map_values, no_exp, number_of_elimination, original_class, scaled_row, strategy, row_number, verbose)
    elif strategy == 'mean':
        important_columns, new_class, new_predict_fn, new_probability, new_row = recreate_row_and_get_new_probabilities_mean(
            feature_ranking, map_values, neutral_points, no_exp, original_class, row_number, verbose)

    if (no_exp == 0 and not (scaled_row == new_row[0]).all()):
        print("Something went wrong! no_exp is 0 but scaled_row is not equal to new_row")

    if (verbose == 1):
        print("Probability: {} Predict_fn: {} Predicted class:{} Actual class:{}"
              .format(original_probability, original_predict_fn, original_class, y_test[row_number]))

        print("Probability: {} Predict_fn: {} Predicted class:{} Actual class:{}"
              .format(new_probability, new_predict_fn, new_class, y_test[row_number]))

    #     map_values = explain_row(new_row[0], verbose)
    # important_columns = [a for (a, b) in map_values[original_class] if b > 0]
    return (
        original_probability[0][0], new_probability[0][0], (original_predict_fn - new_predict_fn)[0][original_class],
        original_class, original_class != new_class, important_columns)


def recreate_row_and_get_new_probabilities_mean(feature_ranking, map_values, neutral_points, no_exp, original_class,
                                                row_number, verbose):
    new_row, important_columns = recreate_row_mean(neutral_points, X_test_unscaled[row_number].copy(), map_values,
                                                   original_class, no_exp, verbose, feature_ranking)
    new_class = model.predict_classes(new_row)[0]
    new_probability = model.predict_proba(new_row)
    new_predict_fn = model.predict(new_row)
    return important_columns, new_class, new_predict_fn, new_probability, new_row


def save_row(row_num, new_class, new_row):
    pass
    with open(newrows_filename, 'a', newline='') as fd:
        writer = csv.writer(fd)
        r = [row_num, new_class]
        r.extend(new_row.tolist())
        writer.writerow(r)


def recreate_row_and_get_new_probabilities_ditribution(feature_ranking, map_values, no_exp, number_of_elimination, original_class,
                                           scaled_row, strategy, row_number, verbose):
    new_probability_list = []
    new_predict_fn_list = []
    number_of_classes = max(y_test) +1

    for i in range(0, number_of_elimination):
        if strategy == "distribution":
            new_row, important_columns = recreate_row_from_distribution_same(scaled_row.copy(), i % number_of_classes, map_values,
                                                                    original_class, no_exp, verbose, feature_ranking)
        elif strategy == "distribution_others":
            new_row, important_columns = recreate_row_from_distribution_other_features(scaled_row.copy(), i % number_of_classes, map_values,
                                                                    original_class, no_exp, verbose, feature_ranking)

        # print ("new prediction for {}: {}".format(i%2, model.predict_proba(new_row)))
        # new_class = model.predict_classes(new_row)[0][0]
        new_probability_list.append(model.predict_proba(new_row))
        new_predict_fn_list.append(model.predict(new_row))

        save_row(row_number, model.predict_classes(new_row)[0], new_row[0])

    new_probability = sum(new_probability_list) / len(new_probability_list)
    new_predict_fn = sum(new_predict_fn_list) / len(new_predict_fn_list)
    new_class = np.argmax(new_probability[0])
    return important_columns, new_class, new_predict_fn, new_probability, new_row


def calculate_values_datapoint(explainer, neutral_points, no_exp, nsamples, results, row_number, verbose,
                               which_explainer, feature_ranking='first', strategy="distribution"):
    start_time = time.time()
    original_probability, new_probability, confidence_diff, original_class, class_change, important_columns = calculate_probability_diff(
        row_number, neutral_points, explainer=explainer, no_exp=no_exp, verbose=verbose,
        which_explainer=which_explainer, nsamples=nsamples, feature_ranking=feature_ranking, strategy=strategy)
    total_time = time.time() - start_time

    # write results to file
    # results.append((original_probability, new_probability, confidence_diff, original_class, class_change,
    #                 no_exp, nsamples, which_explainer, total_time, feature_ranking))
    with open(fileName, 'a', newline='') as fd:
        writer = csv.writer(fd)
        result = [original_probability, new_probability, confidence_diff, original_class, class_change,
                  no_exp, nsamples, which_explainer, total_time, feature_ranking, strategy]
        result.extend(important_columns[0:11])
        writer.writerow(result)


def calculate_values(number_of_rows=200, number_of_exaplanations=11, which_explainer='random', nsampleslist=[100],
                     feature_rankings=['first', 'middle', 'last'], strategies = ["mean", "distribution"], verbose=0):
    explainer = None
    if which_explainer == 'lime':
        explainer = lime_tabular.LimeTabularExplainer(X_train, training_labels=['paid', 'unpaid'])
    elif which_explainer == 'shap':
        explainer = shap.KernelExplainer(model.predict, X_train[0:1000])

    neutral_points = df.mean().drop('class')
    results = []

    predicted_classes = model.predict_classes(X_test)

    correctly_predicted_indices = get_correctly_predicted_indices(number_of_rows)

    counter = 0

    for nsamples in nsampleslist:
        # for row_number in range(number_of_rows):
        for row_number in correctly_predicted_indices:
            print("Row Number {} counter: {}".format(row_number, counter))
            counter = counter + 1
            for no_exp in range(number_of_exaplanations):
                if (predicted_classes[row_number]!= y_test[row_number]):
                    print("Predicted and actual classes are different, skip")
                    continue;

                for feature_ranking in feature_rankings:
                    for strategy in strategies:
                        print("Explanation number:{}---Exaplainer:{}----NSamples:{}---no_exp:{} --feature_ranking:{}---strategy:{}".format(no_exp, which_explainer,
                                                                                            nsamples, no_exp, feature_ranking, strategy))
                        calculate_values_datapoint(explainer, neutral_points, no_exp, nsamples, results, row_number, verbose,
                                           which_explainer, feature_ranking, strategy)

    print(results)
    return results


if os.path.exists(fileName):
    os.remove(fileName)
# create file and add headers
with open(fileName, 'a', newline='') as fd:
    writer = csv.writer(fd)
    writer.writerow(
        ["original_probability", "new_probability", "confidence_diff", "original_class", "class_change", "no_features",
         "nsamples", "explainer", "time", "feature_rankings", "strategy", "i0",
         "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "i10"])


if os.path.exists(newrows_filename):
    os.remove(newrows_filename)
# create file and add headers
with open(newrows_filename, 'a', newline='') as fd:
    writer = csv.writer(fd)
    header = ["row_number", "new_probability"]
    header.extend(["{:02d}".format(x) for x in [*range(shape_size)]])
    writer.writerow(header)

number_of_rows=len(X_test)
strategies =["mean", "distribution", "distribution_others"] #["mean", "distribution", "distribution_others"]
feature_rankings = ['first', 'middle', 'last']

print ("Strating script with explainer: {}".format(sys.argv[1]))

calculate_values(number_of_rows=number_of_rows, number_of_exaplanations=11, which_explainer=sys.argv[1],
                            nsampleslist=["auto"], feature_rankings=feature_rankings, strategies =strategies);

# res_shap = calculate_values(number_of_rows=number_of_rows, number_of_exaplanations=11, which_explainer='shap',
#                             nsampleslist=[100, 1000, "auto"], feature_rankings=feature_rankings, strategies =strategies);
# res_lime = calculate_values(number_of_rows=number_of_rows, number_of_exaplanations=11, which_explainer='lime',
#                             nsampleslist=[100, 1000, "auto"], feature_rankings=feature_rankings, strategies = strategies);
# res_random = calculate_values(number_of_rows=number_of_rows, number_of_exaplanations=11, which_explainer='random',
#                               nsampleslist=["auto"], feature_rankings=feature_rankings, strategies = strategies);
# res_eli5 = calculate_values(number_of_rows=number_of_rows, number_of_exaplanations=11, which_explainer='eli5',
#                             nsampleslist=["auto"], feature_rankings=feature_rankings, strategies =strategies);




# class Animal(object):
#     def __init__(self):
#         pass
#
#     def go_pee(self):
#         print("p")
#
#
#
# a = Animal()