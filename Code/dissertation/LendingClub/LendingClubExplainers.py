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
import eli5
from eli5.sklearn import PermutationImportance

name = 'without_postcode'
fileName = "{}/lending-club-values.csv".format(name)
model = tf.keras.models.load_model('{}/lending-club.h5'.format(name), custom_objects={"f1": kr.f1})
X_train_unscaled = np.load("{}/X_train_unscaled.npy".format(name))
X_test_unscaled = np.load("{}/X_test_unscaled.npy".format(name))
y_train = np.load("{}/y_test.npy".format(name))
y_test = np.load("{}/y_test.npy".format(name))
df = pd.read_csv('{}/lending-club-df.csv'.format(name))
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_unscaled)
X_test = scaler.transform(X_test_unscaled)

prev_scaled_row = None
cached_map_values = None
map_values_eli5 = None
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_unscaled)
X_test = scaler.transform(X_test_unscaled)

shape_size = X_train.shape[1]

# y_test_zero = [y_test.index(a) for a in y_test if a ==0]
# y_test_one = [y_test.index(a) for a in y_test if a ==1]

y_test_zero = np.array(np.where(y_test==0))
y_test_one = np.array(np.where(y_test==1))

def predict_fn(x):
    preds = model.predict(x).reshape(-1, 1)
    p0 = 1 - preds
    return np.hstack((p0, preds))


def explain_row_lime(scaled_row, explainer, nsamples=100, verbose=0):
    if nsamples == "auto":
        exp = explainer.explain_instance(scaled_row, predict_fn, num_features=30,
                                         top_labels=30)  # labels=list(df.columns)
    else:
        exp = explainer.explain_instance(scaled_row, predict_fn, num_features=30, top_labels=30,
                                         num_samples=nsamples)  # labels=list(df.columns)

    if (verbose == 1):
        exp.show_in_notebook(show_table=True, show_all=False)

    return exp.as_map()


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


def explain_row_random(scaled_row, explainer, nsamples=100, verbose=0):
    random_list = {}
    random_list[0] = [(i, random.uniform(0.0, 1.0)) for i in range(shape_size)]
    random_list[1] = [(i, random.uniform(0.0, 1.0)) for i in range(shape_size)]

    map_values = {}
    for class_value in range(len(random_list)):
        s = random_list[class_value]
        sorted_indices = sorted(range(len(s)), key=lambda k: s[k][1], reverse=True)
        #         print(sorted_indices)
        ordered_list = [(a, random_list[class_value][a][1]) for a in sorted_indices if
                        random_list[class_value][a][1] > 0]
        map_values[class_value] = ordered_list
    #     print(map_values)
    return map_values


def explain_row_eli5(scaled_row, explainer, nsamples=100, verbose=0):
    global map_values_eli5
    if map_values_eli5 != None:
        return map_values_eli5

    copy_model = tf.keras.models.load_model('{}/lending-club.h5'.format(name), custom_objects={"f1": kr.f1})

    def base_model():
        return copy_model

    # X_train_unscaled, X_test_unscaled, y_train, y_test
    # train_x, val_x, train_y, val_y
    my_model = KerasRegressor(build_fn=base_model)
    my_model.fit(X_test.copy(), y_test.copy()  )

    perm = PermutationImportance(my_model).fit(X_test[0:1000].copy(), y_test[0:1000].copy())
    # eli5.show_weights(perm, feature_names=list(df.drop('loan_repaid', axis=1).columns))

    s = perm.feature_importances_
    sorted_indices = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
    class_1 = [(a, s[a]) for a in sorted_indices if s[a] > 0]
    sorted_indices = sorted(range(len(s)), key=lambda k: s[k])
    class_0 = [(a, s[a] * -1) for a in sorted_indices if s[a] <= 0]
    map_values_eli5 = {0: class_0, 1: class_1}

    return map_values_eli5


def recreate_row(neutral_points, unscaled_row, map_values, predicted_class, no_exp=3, verbose=0, reverse_order=False):
    tuple_list = map_values[predicted_class]
    if reverse_order:
        important_columns = [a for (a, b) in tuple_list if b > 0][:no_exp]
    else:
        important_columns = [a for (a, b) in tuple_list if b > 0][-no_exp:]

    for c in important_columns:
        unscaled_row[c] = neutral_points[c]

    if (verbose == 1):
        print(pd.DataFrame(unscaled_row))

    return scaler.transform(pd.DataFrame(unscaled_row).values.reshape(1, shape_size))

def recreate_row_from_distribution(scaled_row, which_side, map_values, predicted_class, no_exp=3, verbose=0, reverse_order=False):
    tuple_list = map_values[predicted_class]
    important_columns =[]
    if reverse_order==False:
        important_columns = [a for (a, b) in tuple_list if b > 0][:no_exp]
    elif no_exp>0:
        important_columns = [a for (a, b) in tuple_list if b > 0][-no_exp:]

    for c in important_columns:
        if (which_side == 1) :
            index = y_test_one[0][random.randint(0, y_test_one.shape[1]-1)]
        else:
            index = y_test_zero[0][random.randint(0, y_test_zero.shape[1]-1)]
        scaled_row[c] = X_test[index][c]

    # return scaled_row
    # if (verbose == 1):
    #     print(pd.DataFrame(unscaled_row))

    return pd.DataFrame(scaled_row).values.reshape(1, shape_size)


def calculate_probability_diff(row_number, neutral_points, explainer, no_exp=3, verbose=0, which_explainer='lime',
                               nsamples=100, reverse_order=False, number_of_elimination = 10):
    scaled_row = X_test[row_number].copy()
    # unscaled_row = X_test_unscaled[row_number].copy()

    reshaped_scaled_row = pd.DataFrame(scaled_row).values.reshape(1, shape_size)
    original_class = model.predict_classes(reshaped_scaled_row)[0][0]
    original_probability = model.predict_proba(reshaped_scaled_row)
    original_predict_fn = predict_fn(reshaped_scaled_row)

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
        map_values = explain_row_eli5(scaled_row, explainer, nsamples=nsamples, verbose=verbose)

    prev_scaled_row = scaled_row
    cached_map_values = map_values
    if (verbose == 1):
        print(map_values)

    # new_row = recreate_row(neutral_points, X_test_unscaled[row_number].copy() , map_values, original_class, no_exp, verbose, reverse_order)
    new_probability_list =[]
    new_predict_fn_list = []
    for i in range(0, number_of_elimination):
        new_row = recreate_row_from_distribution(scaled_row.copy(), i%2, map_values, original_class, no_exp, verbose, reverse_order)
        # new_class = model.predict_classes(new_row)[0][0]
        new_probability_list.append( model.predict_proba(new_row))
        new_predict_fn_list.append(predict_fn(new_row))

    new_probability = sum(new_probability_list) / len(new_probability_list)
    new_predict_fn = sum(new_predict_fn_list) / len(new_predict_fn_list)
    new_class =  1 if new_probability >0.5 else 0;

    if (no_exp == 0 and not (scaled_row == new_row[0]).all()):
        print("Something went wrong! no_exp is 0 but scaled_row is not equal to new_row")

    if (verbose == 1):
        print("Probability: {} Predict_fn: {} Predicted class:{} Actual class:{}"
              .format(original_probability, original_predict_fn, original_class, y_test[row_number]))

        print("Probability: {} Predict_fn: {} Predicted class:{} Actual class:{}"
              .format(new_probability, new_predict_fn, new_class, y_test[row_number]))

    #     map_values = explain_row(new_row[0], verbose)
    important_columns = [a for (a, b) in map_values[original_class] if b > 0]
    return (
        original_probability[0][0], new_probability[0][0], (original_predict_fn - new_predict_fn)[0][original_class],
        original_class, original_class != new_class, important_columns)


def calculate_values_datapoint(explainer, neutral_points, no_exp, nsamples, results, row_number, verbose,
                               which_explainer, reverse_order=False):
    start_time = time.time()
    original_probability, new_probability, confidence_diff, original_class, class_change, important_columns = calculate_probability_diff(
        row_number, neutral_points, explainer=explainer, no_exp=no_exp, verbose=verbose,
        which_explainer=which_explainer, nsamples=nsamples, reverse_order=reverse_order)
    total_time = time.time() - start_time

    # write results to file
    results.append((original_probability, new_probability, confidence_diff, original_class, class_change,
                    no_exp, nsamples, which_explainer, total_time, reverse_order))
    with open(fileName, 'a', newline='') as fd:
        writer = csv.writer(fd)
        result = [original_probability, new_probability, confidence_diff, original_class, class_change,
                  no_exp, nsamples, which_explainer, total_time, reverse_order]
        result.extend(important_columns[0:11])
        writer.writerow(result)


def calculate_values(number_of_rows=100, number_of_exaplanations=11, which_explainer='random', nsampleslist=[100],
                     reverse_order=False,
                     verbose=0):
    explainer = None
    if which_explainer == 'lime':
        explainer = lime_tabular.LimeTabularExplainer(X_train, training_labels=['paid', 'unpaid'])
    elif which_explainer == 'shap':
        explainer = shap.KernelExplainer(predict_fn, X_train[0:1000])

    neutral_points = ((df[df['loan_repaid'] != 0].mean() + df[df['loan_repaid'] != 1].mean()) / 2).drop('loan_repaid')
    results = []

    predicted_classes = model.predict_classes(X_test[:number_of_rows])

    for nsamples in nsampleslist:
        for row_number in range(number_of_rows):
            print("Row Number {}".format(row_number))
            for no_exp in range(number_of_exaplanations):
                print("Explanation number:{}---Exaplainer:{}----NSamples:{}".format(no_exp, which_explainer, nsamples))
                if (predicted_classes[row_number][0] != y_test[row_number]):
                    print("Predicted and actual classes are different, skip")
                    continue;

                calculate_values_datapoint(explainer, neutral_points, no_exp, nsamples, results, row_number, verbose,
                                           which_explainer)

                if reverse_order:
                    calculate_values_datapoint(explainer, neutral_points, no_exp, nsamples, results, row_number,
                                               verbose, which_explainer, reverse_order=True)

    print(results)
    return results


import csv
import os

if os.path.exists(fileName):
    os.remove(fileName)
# create file and add headers
with open(fileName, 'a', newline='') as fd:
    writer = csv.writer(fd)
    writer.writerow(
        ["original_probability", "new_probability", "confidence_diff", "original_class", "class_change", "no_features",
         "nsamples", "explainer", "time", "reverse_order", "i0",
         "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "i10"])


number_of_rows=20
res_shap = calculate_values(number_of_rows=number_of_rows, number_of_exaplanations=11, which_explainer='shap',
                            nsampleslist=[100, 1000, "auto"], reverse_order=True);
res_lime = calculate_values(number_of_rows=number_of_rows, number_of_exaplanations=11, which_explainer='lime',
                            nsampleslist=[100, 1000, "auto"],reverse_order=True);
res_random = calculate_values(number_of_rows=number_of_rows, number_of_exaplanations=11, which_explainer='random',
                              nsampleslist=["auto"], reverse_order=True);
res_eli5 = calculate_values(number_of_rows=number_of_rows, number_of_exaplanations=11, which_explainer='eli5',
                            nsampleslist=["auto"], reverse_order=True);


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