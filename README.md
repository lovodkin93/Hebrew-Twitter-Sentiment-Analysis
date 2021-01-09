# Hebrew-Twitter-Sentiment-Analysis

## Install:

```
git clone --recursive https://github.com/lovodkin93/Hebrew-Twitter-Sentiment-Analysis.git
```
## run:

In order to train the model or used a pre-trained model, follow the following steps:
```
cd heb_sentiment_analysis_project
python3 main.py [--flags]
```

possible flags:
1. `--cross-validation` whether to use cross validation. Of Boolean type (default:True).
2. `--print-info` whether to print training information. Of Boolean type (default:True).
3. `--with-stat` whether to print statistics during training. Of Boolean type (default:True).
4. `--save-model` whether to save the trained model (alternative- upload an pre-trained model, which requires to pass the path to that model). Of Boolean type (default:True).
5. `--train-data-path` path to the training data. Of String type (default: `../data/train_morph_data.tsv`).
6. `--test-data-path` path to the test data. Of String type (default: `../data/test_morph_data.tsv`).
7. `--all-data-path` path to the all the data (train and test). Of String type (default: `../data/all_morph_data.tsv`).
8. `--best-no-cv-model-path` path to the pre-trained model that doesn't use cross validation (if `--save-model=False`) or the path where to save the trained model that doesn't use cross validation (if `--save-model=True`). Of String type (default: `best_models/best_no_cv_model.sav`).
9. `--best-cv-model-path` path to the pre-trained model that uses cross validation (if `--save-model=False`) or the path where to save the trained model that uses cross validation (if `--save-model=True`). Of String type (default: `best_models/best_cv_model.sav`).
10. `--best-cv-unbiased-model-path` path to the pre-trained model that uses cross validation (if `--save-model=False`) and that unbiases biased data or the path where to save such trained model (if `--save-model=True`). Of String type (default: `best_models/best_cv_unbiased_model.sav`).
11. `--with-bert` whether to test also the Bert model. Applicable only when `--cross-validation=False`). of Boolean type(default:False).

In addition, if the data has not been tokenized and morphamized yet, it is possible use one of the following morphamizers to pre-process the data:
1. Stanze
2. Yap

Currently, these two tools are the best ones that support the Hebrew language, as current SpaCy doesn't support this language.
Also, in the case that Yap is used, it is necessary to properly install it (for more information, see [Yap](https://github.com/onlplab/yap)), run Yap as RESTful API server (see [Yap](https://github.com/onlplab/yap)), and only when it is connected to the server (might take a few minutes), our model should start working.
In order to pre-process the data, please pass also the following flags:
1. `--morph` determines which pre-process tool to use. Options: Yap, Stanza, None (default:None).
2. `--morphamized-data-train-path` path to save the morphamized train data to. Of String type (default: `../data/train_morph_data.tsv`).
3. `--morphamized-data-test-path` path to save the morphamized test data to. Of String type (default: `../data/test_morph_data.tsv`).
