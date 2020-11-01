import numpy as np

np.random.seed(42)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from utils import CLF
from tamnun.bert import BertClassifier, BertVectorizer
from sklearn.pipeline import make_pipeline

def majority_classifier(test_df, evaluator, majority_label=0):
    predicted_maj = len(test_df) * [majority_label]
    print("The Majority Classifier's results are: ")
    predicted = np.array([np.int64(pred) for pred in predicted_maj])
    evaluator.evaluate(predicted)
    print("\n")
    return


def get_die_value():
    cur_val = np.random.rand()
    if cur_val < 0.5:
        return 0
    else:
        return 1


def throw_a_die_classifier(test_df, evaluator):
    die_predicted = [get_die_value() for _ in range(len(test_df))]
    print("The Throw a die Classifier's results are: ")
    evaluator.evaluate(die_predicted)
    print("\n")


def naive_bayes_classifier(X_train_tf, train_df, X_test_tfidf, evaluator):
    clf = MultinomialNB().fit(X_train_tf, train_df.label)
    predicted = clf.predict(X_test_tfidf)
    predicted = np.array([np.int64(pred) for pred in predicted])
    print("The Naive Bayes Classifier's results are: ")
    mic, mac, cm, acc = evaluator.evaluate(predicted)
    print("\n")

    return clf, mic, mac, cm, acc


def logistic_regression_classifier(X_train_tf, train_df, X_test_tfidf, evaluator):
    clf = LogisticRegression().fit(X_train_tf, train_df.label)
    logistic_predicted = clf.predict(X_test_tfidf)
    logistic_predicted = np.array([np.int64(pred) for pred in logistic_predicted])

    print("The Logistic Regression Classifier's results are: ")

    mic, mac, cm, acc = evaluator.evaluate(logistic_predicted)
    print("\n")

    return clf, mic, mac, cm, acc


def logistic_regression_unibigram_classifier(X_train_tf, train_df, X_test_tfidf, evaluator, help=None):
    if help:
        print(help)
    clf = LogisticRegression().fit(X_train_tf, train_df.label)
    print("The Logistic Regression Classifier's results when looking at unigrams and bigrams are: ")
    logistic_predicted = clf.predict(X_test_tfidf)
    logistic_predicted = np.array([np.int64(pred) for pred in logistic_predicted])

    mic, mac, cm, acc = evaluator.evaluate(logistic_predicted)
    print("\n")

    return clf, mic, mac, cm, acc


def random_forest_classifier(X_train_tf, train_df, X_test_tfidf, evaluator, help=None):
    if help:
        print(help)
    clf = RandomForestClassifier().fit(X_train_tf, train_df.label)
    rf_prediction = clf.predict(X_test_tfidf)
    rf_prediction = np.array([np.int64(pred) for pred in rf_prediction])

    print("The Random Forest Classifier's results are: ")

    mic, mac, cm, acc = evaluator.evaluate(rf_prediction)
    print("\n")

    return clf, mic, mac, cm, acc


def SVM_classifier(X_train_tf, train_df, X_test_tfidf, evaluator, help=None):
    if help:
        print(help)
    clf = SVC(kernel='linear').fit(X_train_tf, train_df.label)
    svm_prediction = clf.predict(X_test_tfidf)
    svm_prediction = np.array([np.int64(pred) for pred in svm_prediction])

    print("The SVM Classifier's results are: ")
    mic, mac, cm, acc = evaluator.evaluate(svm_prediction)
    print("\n")

    return clf, mic, mac, cm, acc


def DecisionTreeRegressor_classifier(X_train_tf, train_df, X_test_tfidf, evaluator, help=None):
    if help:
        print(help)
    clf = DecisionTreeRegressor().fit(X_train_tf, train_df.label)
    rf_prediction = clf.predict(X_test_tfidf)
    rf_prediction = np.array([np.int64(pred) for pred in rf_prediction])

    print("The DecisionTreeRegressor_classifier's results are: ")
    clf, mic, mac, cm, acc=evaluator.evaluate(rf_prediction)
    print("\n")
    return clf, mic, mac, cm, acc




def get_all_Models(size):
    dtr = DecisionTreeRegressor(max_features='auto')
    Mb = MultinomialNB()
    lr_pl2 = LogisticRegression(solver='lbfgs',multi_class='auto')
    lr_no_penelty = LogisticRegression(solver='lbfgs',penalty='none',max_iter=2000,multi_class='auto')
    rf = RandomForestClassifier(n_estimators=size[0])
    svm = SVC(kernel='linear',gamma='scale')
    model_dict = {clf.__class__.__name__: {CLF: clf} for clf in [dtr, Mb,lr_pl2, rf, svm]}

    lr_pl1_name=lr_no_penelty.__class__.__name__+'None'
    model_dict[lr_pl1_name]={CLF: lr_no_penelty}
    model_dict[lr_pl1_name][CLF] = lr_no_penelty
    return model_dict

def Bert_init(num_of_classes=3):
    return make_pipeline(BertVectorizer(), BertClassifier(num_of_classes=num_of_classes))