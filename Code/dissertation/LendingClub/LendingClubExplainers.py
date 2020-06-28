import tensorflow as tf
import shap
from lime import lime_tabular
import numpy as np
import random
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler

name = 'with_postcode'
fileName = "{}/lending-club-values.csv".format(name)
model = tf.keras.models.load_model('{}/lending-club.h5'.format(name))
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
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_unscaled)
X_test = scaler.transform(X_test_unscaled)

shape_size = X_train.shape[1]


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


def recreate_row(neutral_points, unscaled_row, map_values, predicted_class, no_exp=3, verbose=0):
    tuple_list = map_values[predicted_class]
    important_columns = [a for (a, b) in tuple_list if b > 0][:no_exp]

    for c in important_columns:
        unscaled_row[c] = neutral_points[c]

    if (verbose == 1):
        print(pd.DataFrame(unscaled_row))

    return scaler.transform(pd.DataFrame(unscaled_row).values.reshape(1, shape_size))


def calculate_probability_diff(row_number, neutral_points, explainer, no_exp=3, verbose=0, which_explainer='lime',
                               nsamples=100):
    scaled_row = X_test[row_number]
    unscaled_row = X_test_unscaled[row_number].copy()

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

    prev_scaled_row = scaled_row
    cached_map_values = map_values
    if (verbose == 1):
        print(map_values)

    new_row = recreate_row(neutral_points, unscaled_row, map_values, original_class, no_exp, verbose)
    new_class = model.predict_classes(new_row)[0][0]
    new_probability = model.predict_proba(new_row)
    new_predict_fn = predict_fn(new_row)

    if (no_exp == 0 and (scaled_row != new_row).all()):
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


def calculate_values(number_of_rows=100, number_of_exaplanations=11, which_explainer='random', nsampleslist=[100],
                     verbose=0):
    if which_explainer == 'lime':
        explainer = lime_tabular.LimeTabularExplainer(X_train, training_labels=['paid', 'unpaid'])
    elif which_explainer == 'shap':
        explainer = shap.KernelExplainer(predict_fn, X_train[0:1000])
    elif which_explainer == 'random':
        explainer = None

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

                start_time = time.time()
                original_probability, new_probability, confidence_diff, original_class, class_change, important_columns = calculate_probability_diff(
                    row_number, neutral_points, explainer=explainer, no_exp=no_exp, verbose=verbose,
                    which_explainer=which_explainer, nsamples=nsamples)
                total_time = time.time() - start_time
                results.append((original_probability, new_probability, confidence_diff, original_class, class_change,
                                no_exp, nsamples, which_explainer, total_time))
                with open(fileName, 'a', newline='') as fd:
                    writer = csv.writer(fd)
                    result = [original_probability, new_probability, confidence_diff, original_class, class_change,
                              no_exp, nsamples, which_explainer, total_time]
                    result.extend(important_columns[0:11])
                    writer.writerow(result)

    print(results)
    return results


import csv
import os

if os.path.exists(fileName):
    os.remove(fileName)
# add headers
with open(fileName, 'a', newline='') as fd:
    writer = csv.writer(fd)
    writer.writerow(
        ["original_probability", "new_probability", "confidence_diff", "original_class", "class_change", "no_features",
         "nsamples", "explainer", "time", "important_columns",
         "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "i10"])

res_shap = calculate_values(number_of_rows=200, number_of_exaplanations=11, which_explainer='shap',
                            nsampleslist=[100, 1000, "auto"]);
res_lime = calculate_values(number_of_rows=200, number_of_exaplanations=11, which_explainer='lime',
                            nsampleslist=[100, 1000, "auto"]);
res_random = calculate_values(number_of_rows=200, number_of_exaplanations=11, which_explainer='random',
                              nsampleslist=[100, 1000, "auto"]);
