from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from train import train_all_models, get_best_model, train_all_models_with_cv, train_all_models_with_cv_balance, \
    data_pre_processing, data_acquisition
from morphamizer import create_morphamaized_file
from utils import get_strongest_words
import utils
import morphamizer as morph
from config import Config
import sys

import pickle

def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


def load_model(filename):
    return pickle.load(open(filename, 'rb'))

def predict_preprocess(df, range=(1, 1)):
    clean_df = morph.morph_df(morph.get_clean_data(df), tosave=False, morph_path=False)
    count_vect = CountVectorizer(ngram_range=range)
    X_counts = count_vect.fit_transform(clean_df.text)

    tf_transformer = TfidfTransformer(use_idf=False).fit(X_counts)
    X_tf = tf_transformer.transform(X_counts)

    return X_tf


def train_model(config):
    if config.args.cross_validation:
        measurement = utils.ACCURACY
        print()
        print( "############ unbias data ############")
        print()
        models_dict_balanced, evaluator_balanced = train_all_models_with_cv_balance(config)
        best_model_b, name_b, pred_b, score_b = get_best_model(models_dict_balanced, None, measurement)
        print()
        print("############ original data ############")
        print()
        models_dict, count_vect, evaluator = train_all_models_with_cv(config)
        best_model, name, pred, score = get_best_model(models_dict, None, measurement)


    else:
        models_dict, count_vect, evaluator = train_all_models(config)
        measurement = utils.F1
        best_model, name, pred, score = get_best_model(models_dict, utils.MICRO, measurement)
    if config.args.save_model:
        if config.args.cross_validation:
            save_model(best_model, config.args.best_cv_model_path)
            save_model(best_model_b, config.args.best_cv_unbiased_model_path)
        else:
            save_model(best_model, config.args.best_no_cv_model_path)

    else:
        if config.args.cross_validation:
            if score_b < score:
                best_model = load_model(config.args.best_cv_model_path)
            else:
                best_model = load_model(config.args.best_cv_unbiased_model_path)
        else:
            best_model = load_model(config.args.best_no_cv_model_path)
    if config.args.print_info:
        evaluator.show_errors(pred, name)
        evaluator.show_correct(pred, name)
    return best_model


if __name__ == "__main__":
    # main()
    model = train_model(config=Config())
