import os
import sys
import numpy as np
import pandas as pd
import operator
from time import time
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

# from sklearn.neural_network import MLPClassifier
from probabilityDistribution import get_probabilities, ProbabilityDistribution, ConditionalProbabilityDistribution


def measure_time(func):
    def my_decorator(*args):
        t = time()
        ret = func(*args)
        t -= time()
        sys.stdout.write('{0} function took {1:.3f} seconds\n'.format(func.__name__, -t))
        return ret

    return my_decorator


@measure_time
def read_data(path, filename):
    # Read data and store it into a dataframe
    df = pd.read_csv(os.path.join(path, filename), parse_dates=[["Date", "Time"]])
    print(df)
    df.index = df['Date_Time']
    del df["Date_Time"]

    # Normalize some variables
    for var in ['Open', 'High', 'Low', 'Parab 0.02', 'media 10', 'media 50', 'media 100',
                'media 200', 'Xmedia 10', 'Xmedia 50', 'Xmedia 100', 'Xmedia 200', 'Hmedia 10',
                'Hmedia 50', 'Hmedia 100', 'Hmedia 200', 'MAXIMOS10', 'MAXIMOS30', 'MINIMOS10',
                'MINIMOS30', 'mediaH 200', 'XmediaH 10', 'XmediaH 50', 'XmediaH 100', 'XmediaH 200', 'HmediaH 10',
                'HmediaH 50', 'HmediaH 100', 'HmediaH 200', 'mediaL 200', 'XmediaL 10', 'XmediaL 50', 'XmediaL 100',
                'XmediaL 200', 'HmediaL 10', 'HmediaL 50', 'HmediaL 100', 'HmediaL 200']:
        df[var] /= df["Close"]
        df[var] -= 1.
        df[var] *= 100

    return df


@measure_time
def turn_floats_into_categories(df):
    # Define categories
    cuts = {}
    nCategories = 4
    targets = ["TAR5", "TAR10", "TAR24", "Close", "DAYOFWEEK"]
    for a in df:
        if str(a) not in targets:
            print(a)
            paso = (df[a].max() - df[a].min()) / nCategories
            cuts.update({a: [-np.inf, paso, 2 * paso, 3 * paso, np.inf]})

    # Substitute float values by labels
    labels = {}
    print(cuts.keys())
    for variable in cuts.keys():
        these_cuts = cuts[variable]
        values = df[variable].values
        labels[variable] = []
        for index in range(len(these_cuts) - 1):
            a, b = these_cuts[index], these_cuts[index + 1]
            label = "{0:.2e} < x <= {1:.2e}".format(a, b)
            labels[variable].append(label)
            mask = np.logical_and(values >= a, values <= b)
            df.loc[mask, variable] = label

    for variable in cuts.keys():
        df[variable] = df[variable].replace(labels[variable], range(len(labels[variable])))
    print(cuts)
    print(df)
    return df


def iamb(probabilities, target, eps_g=1e-5, eps_s=1e-4):
    """
    Calculates the Markov Blanket of the target variable using the Incremantal Association Markov
     Blanket algorithm

    Parameters
    ----------
    probabilities : ProbabilityDistribution Object
                    joint probabilities of all the considered variables
    target : string
             targeted variable whose MB is going to be calculated
    eps_g : float
            threshold used in the Growing phase
    eps_s : float
            threshold used in the Shrinking phase

    Returns
    -------
    markov_blanket : set
                     labels of the variables that form the Markov Blanket
    """

    # Initializes some variables
    t = [target]
    markov_blanket = set([])
    variables = set(probabilities.variables) - set(t)

    # Growing phase
    while len(variables) != 0:
        # Evaluate all conditional mutual information
        cmi = {}
        if len(markov_blanket) == 0:
            for variable in variables:
                cmi[variable] = probabilities.get_mutual_information(t, [variable])
        else:
            for variable in variables:
                cmi[variable] = probabilities.get_conditional_mutual_information(t, [variable],
                                                                                 list(markov_blanket))
        # Add variable that provides maximum CMI if its value is higher than some threshold
        label, value = max(cmi.items(), key=operator.itemgetter(1))
        if value > eps_g:
            markov_blanket = markov_blanket.union([label])
            variables = variables - set([label])
        else:
            break

    # Shrinking phase
    for variable in markov_blanket:
        cmi = probabilities.get_conditional_mutual_information(t, [variable],
                                                               list(markov_blanket - set(variable)))
        if cmi < eps_s:
            markov_blanket = markov_blanket - set(variable)

    return markov_blanket


def iamb2(df, vars, target, eps_g=1e-5, eps_s=1e-4):
    """
    Calculates the Markov Blanket of the target variable using the Incremantal Association Markov
     Blanket algorithm

    Parameters
    ----------
    probabilities : ProbabilityDistribution Object
                    joint probabilities of all the considered variables
    target : string
             targeted variable whose MB is going to be calculated
    eps_g : float
            threshold used in the Growing phase
    eps_s : float
            threshold used in the Shrinking phase

    Returns
    -------
    markov_blanket : set
                     labels of the variables that form the Markov Blanket
    """

    # Initializes some variables
    t = [target]
    markov_blanket = set([])
    variables = set(vars) - set(t)
    max_features = 6

    # Growing phase

    while len(variables) != 0 and len(markov_blanket) < max_features:
        # Evaluate all conditional mutual information
        cmi = {}
        if len(markov_blanket) == 0:
            for variable in variables:
                pdist = get_probabilities(df, [variable, target])
                cmi[variable] = pdist.get_mutual_information(t, [variable])
                print(variable, cmi[variable])
        else:
            for variable in variables:
                pdist2 = get_probabilities(df, list(markov_blanket) + [variable] + [target])
                cmi[variable] = pdist2.get_conditional_mutual_information(t, [variable], list(markov_blanket))
                print(variable, cmi[variable])
        # Add variable that provides maximum CMI if its value is higher than some threshold
        label, value = max(cmi.items(), key=operator.itemgetter(1))
        if value > eps_g:
            markov_blanket = markov_blanket.union([label])
            print('MB', markov_blanket)
            variables = variables - set([label])
        else:
            break

    # Shrinking phase
    for variable in markov_blanket:
        pdist = get_probabilities(df, list(markov_blanket) + [variable] + [target])
        cmi = pdist.get_conditional_mutual_information(t, [variable], list(markov_blanket - set(variable)))
        if cmi < eps_s:
            markov_blanket = markov_blanket - set(variable)

    return markov_blanket

    return CMO


@measure_time
def find_optimal_predictors(df):
    # features=df.keys()-["TAR5","TAR10","TAR24","Close"]
    features = df.keys().drop(["TAR5", "TAR10", "TAR24", "Close"], 1)

    return iamb2(df, features, "TAR10", 1e-5, 1e-6)


@measure_time
def train_predictor(df, markov_blanket, p_train=0.6):
    # DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
    #                        min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
    #                        min_impurity_split=1e-07, class_weight=None, presort=False)


    # RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
    #                        min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
    #                        min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None,
    #                        verbose=0, warm_start=False, class_weight=None)





    rf = RandomForestClassifier(n_estimators=5)
    clf1 = tree.DecisionTreeClassifier(max_leaf_nodes=10, class_weight=None)

    x = df[list(markov_blanket)].values
    y = df["TAR10"].values

    n_samples = x.shape[0]
    n_train = int(np.round(p_train * n_samples))
    xt = x[:n_train, :]
    yt = y[:n_train]

    n_check = n_samples - n_train
    xc = x[n_train:, :]
    yc = y[n_train:]
    ynames = ["lateral", "alcista"]
    xnames = list(markov_blanket)
    clf1.fit(xt, yt)
    sys.stdout.write("Result INS is {}\n".format(clf1.score(xt, yt)))
    sys.stdout.write("Result OOS is {}\n".format(clf1.score(xc, yc)))
    scores = confusion_matrix(yt, clf1.predict(xt), labels=[0, 1, ])
    scores2 = confusion_matrix(yc, clf1.predict(xc), labels=[0, 1])
    print(scores)
    print(scores2)
    tree.export_graphviz(clf1, out_file='Tree.dot', class_names=ynames,
                         feature_names=xnames)
    return rf


def main(path, filename):
    # print(df)
    # Read data and pre-process it
    df = read_data(path, filename)
    print(df)

    # Turn continuous variables into discrete categories
    df = turn_floats_into_categories(df)
    print(df)

    # Select variables to use
    markov_blanket = find_optimal_predictors(df)
    print(df[list(markov_blanket)].values)

    # Train model and verify its accuracy
    predictor = train_predictor(df, markov_blanket)


if __name__ == "__main__":
    main("./", "CLdata.csv")
