from pathlib import Path
from evaluate_sentiment import Evaluator
import pandas as pd
from classifiers import *
import utils
import morphamizer as morph
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE
from sklearn.feature_selection import chi2

MORPH_TRAIN_PATH = 'data/train_tweet_data_labeld_final_morph_yap.tsv' # MORPH_TRAIN_PATH = r'/cs/labs/oabend/lovodkin93/workspace/project_submission5/data/train_tweet_data_labeld_final_morph_yap.tsv' #path to the morphamized tsv
MORPH_TEST_PATH = 'data/test_tweet_data_labeld_final_morph_yap.tsv' # MORPH_TEST_PATH = r'/cs/labs/oabend/lovodkin93/workspace/project_submission5/data/test_tweet_data_labeld_final_morph_yap.tsv' #path to the morphamized tsv
MORPH_ALL_PATH = 'data/all_tweet_data_labeld_final_morph_yap.tsv' # MORPH_ALL_PATH = r'/cs/labs/oabend/lovodkin93/workspace/project_submission5/data/all_tweet_data_labeld_final_morph_yap.tsv' # path to the morphamized tsv

def data_balance(train_df, test_df=None, ngram=(1, 1)):
    from collections import Counter
    # count_vect = CountVectorizer(ngram_range=ngram,max_features=2500)
    count_vect = CountVectorizer(ngram_range=ngram)
    y_tr = train_df.label
    # y_tr = y_tr.astype(int)
    X_train_counts = count_vect.fit_transform(train_df.text)

    smk_tr = SMOTETomek()
    X_train_counts, y_tr_res = smk_tr.fit_sample(X_train_counts, y_tr)
    print(f'original  data set count{Counter(y_tr)}')
    print(f'new balanced data set count{Counter(y_tr_res)}')
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    if test_df is not None:
        y_ts = test_df.label
        # y_ts = y_ts.astype(int)
        X_test_counts = count_vect.transform(test_df.text)
        smk_ts = SMOTETomek()
        x_ts_res, y_ts_res = smk_ts.fit_sample(X_test_counts, y_ts)
        tf_transformer = TfidfTransformer(use_idf=False).fit(X_test_counts)
        X_test_tfidf = tf_transformer.transform(X_test_counts)

        print(f'original ts ds count{Counter(y_ts)}')
        print(f'new st ds count{Counter(y_ts_res)}')
        return X_train_counts, X_train_tf, y_tr_res, X_test_counts, X_test_tfidf, y_ts
    return X_train_counts, X_train_tf, y_tr_res


def ngram_text_tf_idf(train_txt, test_text, range=(1, 2)):
    count_vect = CountVectorizer(ngram_range=range)

    X_train_counts = count_vect.fit_transform(train_txt)

    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)

    X_test_counts = count_vect.transform(test_text)
    X_test_tfidf = tf_transformer.transform(X_test_counts)

    return X_train_tf, X_test_tfidf


def data_pre_processing(train_text, test_text):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_text)

    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)

    X_test_counts = count_vect.transform(test_text)
    X_test_tfidf = tf_transformer.transform(X_test_counts)

    return X_train_counts, X_test_counts, X_train_tf, X_test_tfidf, count_vect

#

def data_acquisition(train_path, test_path, to_morph, print_info=True, with_stat=True):
    train_path = Path(train_path)
    test_path = Path(test_path)
    test_raw = pd.read_csv(test_path, sep="\t", encoding="utf-8", names=["text", "label"])
    train_raw = pd.read_csv(train_path, sep="\t", encoding="utf-8", names=["text", "label"])
    train_df = morph.get_clean_data(train_raw)
    test_df = morph.get_clean_data(test_raw)
    if to_morph:
        train_df = morph.yap_morph_df(train_df, tosave=True,
                                      morph_path=MORPH_TRAIN_PATH)  # yap_morph_def in stead of morph_df
        test_df = morph.yap_morph_df(test_df, tosave=True,
                                     morph_path=MORPH_TEST_PATH)  # yap_morph_def in stead of morph_df

    train_df.loc[:, "label"] = train_df.label.astype("category")
    text = train_df["text"]
    if print_info:
        utils.print_data_layout(train_df)
    if with_stat:
        utils.plot_character_length_histogram(text)
        utils.plot_word_number_histogram(train_df.text)
        stop = utils.get_hebrew_stopwords()
        utils.plot_top_non_stopwords_barchart(text, stop)
        utils.plot_top_non_stopwords_barchart(train_df[train_df.label == 0].text, stop, "Positive top words")
        utils.plot_top_non_stopwords_barchart(train_df[train_df.label == 1].text, stop, "Negative top words")
        utils.plot_top_non_stopwords_barchart(train_df[train_df.label == 2].text, stop, "Neutral top words")

    return train_df, test_df


def train_all_models(train_path, test_path, to_morph, print_info, with_stat):
    train_df, test_df = data_acquisition(train_path, test_path, to_morph, print_info, with_stat)

    X_train_counts, X_test_counts, X_train_tf, X_test_tfidf, count_vect = data_pre_processing(train_df.text,
                                                                                              test_df.text)

    evaluator = Evaluator(test_path, test_df)

    # trying out different classifiers

    # we won't save these they are for baseline
    majority_classifier(test_df, evaluator, 0)
    throw_a_die_classifier(test_df, evaluator)

    model_dict = get_all_Models(size=test_df.shape)
    count = 0
    for model in model_dict:
        clf, mic, mac, cm, acc, pred = train_eval_model(model_dict[model][utils.CLF], model, X_train_tf, train_df.label,
                                                        X_test_tfidf,
                                                        evaluator, feature_selection=False)
        model_dict = utils.update_dict(model_dict, model, clf, mic, mac, cm, acc, pred)

        count += 1
    ngrams = [(1, 2), (1, 3)]
    for ngram in ngrams:
        modles = get_all_Models(size=test_df.shape)

        X_train_tf, X_test_tfidf = ngram_text_tf_idf(train_df.text, test_df.text, ngram)

        for model in modles:
            model_name = model + str(ngram)
            clf, mic, mac, cm, acc, pred = train_eval_model(modles[model][utils.CLF], model_name, X_train_tf,
                                                            train_df.label,
                                                            X_test_tfidf, evaluator, feature_selection=False)
            model_dict[model_name] = {utils.CLF: clf}
            model_dict = utils.update_dict(model_dict, model_name, clf, mic, mac, cm, acc, pred)

            count += 1
    # todo your bert
    from datetime import datetime
    start_time = datetime.now()

    bert = Bert_init()
    model_name = 'bert'
    clf, mic, mac, cm, acc, pred = train_eval_model(clf=bert, name=model_name, X_train_tf=train_df.text,
                                                    label=train_df.label, X_test_tfidf=test_df.text, evaluator=evaluator)
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    model_dict[model_name] = {utils.CLF: clf}
    model_dict = utils.update_dict(model_dict, model_name, clf, mic, mac, cm, acc, pred)



    count += 1
    print(f"# of models ={count}")

    return model_dict, count_vect, evaluator


def cv_data_pre_processing(train_text):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_text)
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    return X_train_counts, X_train_tf, count_vect


def cv_data_acquisition(train_path, to_morph, print_info, with_stat):
    train_path = Path(train_path)
    train_raw = pd.read_csv(train_path, sep="\t", encoding="utf-8", names=["text", "label"])
    train_df = morph.get_clean_data(train_raw)

    if to_morph:
        train_df = morph.yap_morph_df(train_df, tosave=True,
                                      morph_path=MORPH_TRAIN_PATH)  # yap_morph_def in stead of morph_df
    train_df.loc[:, "label"] = train_df.label.astype("category")
    text = train_df["text"]
    if print_info:
        utils.print_data_layout(train_df)
    if with_stat:
        utils.plot_character_length_histogram(text)
        utils.plot_word_number_histogram(train_df.text)
        stop = utils.get_hebrew_stopwords()
        utils.plot_top_non_stopwords_barchart(text, stop)
        utils.plot_top_non_stopwords_barchart(train_df[train_df.label == 0].text, stop, "Positive top words")
        utils.plot_top_non_stopwords_barchart(train_df[train_df.label == 1].text, stop, "Negative top words")
    return train_df


def train_all_models_with_cv_balance(train_path, test_path, all_path, to_morph, print_info, with_stat, k=10):
    train_df, test_df = data_acquisition(train_path, test_path, to_morph, print_info, with_stat)
    cv_df = cv_data_acquisition(all_path, to_morph, print_info, with_stat)
    cv = KFold(n_splits=k, random_state=1, shuffle=True)
    # cv_evaluator = Evaluator(test_path, cv_df)

    evaluator = Evaluator(test_path, test_df)
    # we won't save these they are for baseline
    majority_classifier(test_df, evaluator, 0)
    throw_a_die_classifier(test_df, evaluator)

    count = 0
    model_dict = {}

    ngrams = [(1, 1), (1, 2), (1, 3)]
    for ngram in ngrams:

        cv_X_train_counts, cv_X_train_tf, y_cv_res = data_balance(cv_df)
        X_train_tf, X_test_tfidf, y_tr_res , y_ts_res =train_test_split(cv_X_train_tf, y_cv_res)
        # X_train_tf, X_test_tfidf = ngram_text_tf_idf(train_df.text, test_df.text, ngram)
        modles = get_all_Models(size=cv_X_train_counts.shape)
        # modles = {}
        evaluator.set_label(y_ts_res)

        for model in modles:
            model_name = model + str(ngram)

            clf, mic, mac, cm, acc, pred = train_eval_model(modles[model][utils.CLF], model_name, X_train_tf,
                                                            y_tr_res,
                                                            X_test_tfidf, evaluator, feature_selection=False)

            model_dict[model_name] = {utils.CLF: clf}
            model_dict = utils.update_dict(model_dict, model_name, clf, mic, mac, cm, acc, pred)
            scores = cross_val_score(model_dict[model_name][utils.CLF], cv_X_train_tf,y_cv_res,cv=cv, n_jobs=-1)
            mean_cv = scores.mean()
            model_dict[model_name][utils.CROSS_VALIDATION] = mean_cv
            count += 1
            # print(f'mean Cross validation : {mean_cv}')
            print("\n")

    print(f"# of models ={count} with balancing")
    if print_info:
        for model in model_dict:
            print(f" model = {model}", end=' ')
            print(f"cv={model_dict[model][utils.CROSS_VALIDATION]}")
    print("\n")
    return model_dict, evaluator


def train_all_models_with_cv(train_path, test_path, all_path, to_morph, print_info, with_stat, k=10):
    # train test data
    train_df, test_df = data_acquisition(train_path, test_path, to_morph, print_info, with_stat)

    X_train_counts, X_test_counts, X_train_tf, X_test_tfidf, count_vect = data_pre_processing(train_df.text,
                                                                                              test_df.text)

    cv_df = cv_data_acquisition(all_path, to_morph, print_info, with_stat)
    cv_X_train_counts, cv_X_train_tf, cv_count_vect = cv_data_pre_processing(cv_df.text)

    cv = KFold(n_splits=k, random_state=1, shuffle=True)
    evaluator = Evaluator(test_path, test_df)

    # trying out different classifiers

    # we won't save these they are for baseline
    majority_classifier(test_df, evaluator, 0)
    throw_a_die_classifier(test_df, evaluator)

    model_dict={}
    count = 0
    ngrams = [(1, 1), (1, 2), (1, 3)]

    for ngram in ngrams:
        modles = get_all_Models(size=test_df.shape)

        X_train_tf, X_test_tfidf = ngram_text_tf_idf(train_df.text, test_df.text, ngram)

        for model in modles:
            model_name = model + str(ngram)
            clf, mic, mac, cm, acc, pred = train_eval_model(modles[model][utils.CLF], model_name, X_train_tf,
                                                            train_df.label,
                                                            X_test_tfidf, evaluator, feature_selection=False)
            model_dict[model_name] = {utils.CLF: clf}
            model_dict = utils.update_dict(model_dict, model_name, clf, mic, mac, cm, acc, pred)

            scores = cross_val_score(model_dict[model_name][utils.CLF], cv_X_train_tf, cv_df.label, scoring='accuracy',
                                     cv=cv, n_jobs=-1)

            mean_cv = np.mean(scores)
            model_dict[model_name][utils.CROSS_VALIDATION] = mean_cv
            count += 1
            # print(f'mean Cross validation : {mean_cv}')
            print("\n")

    print(f"# of models ={count} without balancing")
    if print_info:
        for model in model_dict:
            print(f" model = {model}", end=' ')
            print(f"cv={model_dict[model][utils.CROSS_VALIDATION]}")
    print("\n")
    return model_dict, count_vect, evaluator


def train_eval_model(clf, name, X_train_tf, label, X_test_tfidf, evaluator, feature_selection=False, help=None):
    
    if feature_selection :
        rfe = RFE(clf, n_features_to_select=10)
        fit = rfe.fit_transform(X_train_tf,label)
        prediction = rfe.predict(X_test_tfidf)
        prediction = np.array([np.int64(pred) for pred in prediction])
        if help:
            print("Num Features: %s" % (fit.n_features_))
            print("Selected Features: %s" % (fit.support_))
            print("Feature Ranking: %s" % (fit.ranking_))

        print(f"The{name}'s results are: ")
        mic, mac, cm, acc = evaluator.evaluate(prediction)
        return rfe, mic, mac, cm, acc, prediction
    clf.fit(X_train_tf, label)

    prediction = clf.predict(X_test_tfidf)
    prediction = np.array([np.int64(pred) for pred in prediction])

    print(f"The{name}'s results are: ")
    mic, mac, cm, acc = evaluator.evaluate(prediction)

    return clf, mic, mac, cm, acc, prediction

def get_best_model(model_dict, micro_macro=utils.MICRO, measurement=utils.F1):
    """ find best model under mesuement """
    target = 0
    best = None
    name = None
    if micro_macro == None:

        for model in model_dict:
            if model_dict[model][measurement] > target:
                target = model_dict[model][measurement]
                best = model_dict[model]
                name = model
        pred = model_dict[name][utils.PREDICTION]
    else:
        for model in model_dict:
            if model_dict[model][micro_macro][measurement] > target:
                target = model_dict[model][micro_macro][measurement]
                best = model_dict[model]
                name = model
        pred = model_dict[name][utils.PREDICTION]
    return best, name, pred, target
