from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from train import train_all_models, get_best_model, train_all_models_with_cv, train_all_models_with_cv_balance, \
    data_pre_processing, data_acquisition
from morphamizer import create_morphamaized_file
from utils import get_strongest_words
import utils
import morphamizer as morph
import sys

BEST_CV_MODEL_PATH = r'best_models\finalized_tweet_cv_model_without_balanced_data.sav' # BEST_CV_MODEL_PATH = r'/cs/labs/oabend/lovodkin93/workspace/project_submission5/best_models/finalized_tweet_cv_model_without_balanced_data.sav'
BEST_UNBIASED_DATA_CV_MODEL_PATH = r'best_models\finalized_tweet_cv_model_with_balanced_data.sav' # BEST_UNBIASED_DATA_CV_MODEL_PATH = r'/cs/labs/oabend/lovodkin93/workspace/project_submission5/best_models/finalized_tweet_cv_model_with_balanced_data.sav'
BEST_MODEL_PATH = r'best_models\finalized_tweet_model.sav' # BEST_MODEL_PATH = r'/cs/labs/oabend/lovodkin93/workspace/project_submission5/best_models/finalized_tweet_model.sav'

TEST_PATH = r'data\test_tweet_data_labeld_final_morph_yap.tsv' # TEST_PATH = r'/cs/labs/oabend/lovodkin93/workspace/project_submission5/data/test_tweet_data_labeld_final.tsv'# if morphamizing then change this one to the "... final.tsv" (the before morphamization).

TRAIN_PATH = r'data\train_tweet_data_labeld_final_morph_yap.tsv' # TRAIN_PATH = r'/cs/labs/oabend/lovodkin93/workspace/project_submission5/data/train_tweet_data_labeld_final.tsv' #if morphamizing then change this one to the "... final.tsv" (the before morphamization).

ALL_PATH = r'data\all_tweet_data_labeld_final_morph_yap.tsv' #ALL_PATH = r'/cs/labs/oabend/lovodkin93/workspace/project_submission5/data/all_tweet_data_labeld_final_morph_yap.tsv'


import pickle


def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


def load_model(filename=BEST_MODEL_PATH):
    return pickle.load(open(filename, 'rb'))

def predict_preprocess(df, range=(1, 1)):
    clean_df = morph.morph_df(morph.get_clean_data(df), tosave=False, morph_path=False)
    count_vect = CountVectorizer(ngram_range=range)
    X_counts = count_vect.fit_transform(clean_df.text)

    tf_transformer = TfidfTransformer(use_idf=False).fit(X_counts)
    X_tf = tf_transformer.transform(X_counts)

    return X_tf


def train_model(to_morph=False, print_info=True, with_stat=True, save_model_f=False, cv=True):
    train_path = TRAIN_PATH
    test_path = TEST_PATH
    if cv:
        measurement = utils.ACCURACY
        print()
        print( "############ unbias data############")
        print()
        models_dict_balanced, evaluator_balanced = train_all_models_with_cv_balance(train_path, test_path, ALL_PATH, to_morph,
                                                                  print_info, with_stat)
        best_model_b, name_b, pred_b, score_b = get_best_model(models_dict_balanced, None, measurement)
        print()
        print("############ original data   ############")
        print()
        models_dict, count_vect, evaluator = train_all_models_with_cv(train_path, test_path, ALL_PATH, to_morph,
                                                                      print_info, with_stat)
        best_model, name, pred, score = get_best_model(models_dict, None, measurement)


    else:
        models_dict, count_vect, evaluator = train_all_models(train_path, test_path, to_morph, print_info,
                                                              with_stat)
        measurement = utils.F1
        best_model, name, pred, score = get_best_model(models_dict, utils.MICRO, measurement)
    if save_model_f:
        if cv:
            save_model(best_model, BEST_CV_MODEL_PATH)
            save_model(best_model_b, BEST_UNBIASED_DATA_CV_MODEL_PATH)
        else:
            save_model(best_model, BEST_MODEL_PATH)

    else:
        if cv:
            if score_b < score:
                best_model = load_model(BEST_CV_MODEL_PATH)
            else:
                best_model = load_model(BEST_UNBIASED_DATA_CV_MODEL_PATH)
        else:
            best_model = load_model(BEST_MODEL_PATH)
    if print_info:

        evaluator.show_errors(pred, name)
        evaluator.show_correct(pred, name)

    return best_model


if __name__ == "__main__":
    # main()
    model = train_model(cv=True, to_morph=False, print_info=True, with_stat=True, save_model_f=True)
