import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

# HEB_STOP_WORD_PATH = 'data/heb_stopwords.txt'
HEB_STOP_WORD_PATH = 'data/heb_legal_stopwords.txt'
F1 = 'f1'
PRECISION = 'precision'
RECALL = 'recall'
MICRO = 'micro'
MACRO = 'macro'
CONFUSION_MATRIX = 'confusion_matrix'
CLF = 'classifier'
ACCURACY = 'accuracy'
PREDICTION = 'prediction'
CROSS_VALIDATION = 'cross_val'


def plot_top_non_stopwords_barchart(text, stop, title='top_non_stopwords_barchart'):
    # stop=set(get_hebrew_stopwords())

    new = text.str.split()
    new = new.values.tolist()
    corpus = [word for i in new for word in i]

    counter = Counter(corpus)
    most = counter.most_common()
    x, y = [], []
    for word, count in most[:50]:
        if (len(word) > 1 and word not in stop):
            x.append(word)
            y.append(count)

    sns.barplot(x=y, y=invert_words(x))
    plt.title(title)
    plt.show()


def invert_words(words):
    return [w[::-1] for w in words]


def get_hebrew_stopwords(stop_path=HEB_STOP_WORD_PATH):
    with open(stop_path, encoding="utf-8") as in_file:
        lines = in_file.readlines()
        res = [l.strip() for l in lines]
        print(res[:4])
    return res


def plot_word_number_histogram(text):
    text.str.split().map(lambda x: len(x)).hist(range=(0, 300))
    plt.title('word_number_histogram')
    plt.show()


def plot_character_length_histogram(text):
    text.str.len().hist(range=(0, 2000))
    plt.title('character_length_histogram')
    plt.show()


def print_data_layout(train_df):
    tot = len(train_df)
    print(train_df.label.value_counts())
    print(train_df.label.value_counts() / tot)
    sns.countplot(x=train_df.label, data=train_df)
    plt.show()


def get_strongest_words(clf, count_vect):
    inverse_dict = {count_vect.vocabulary_[w]: w for w in count_vect.vocabulary_.keys()}

    for lbl in range(len(clf.coef_)):
        cur_coef = clf.coef_[lbl]
        word_df = pd.DataFrame({"val": cur_coef}).reset_index().sort_values(["val"], ascending=[False])

        word_df.loc[:, "word"] = word_df["index"].apply(lambda v: inverse_dict[v])
        print(f'best words for label {lbl}')
        print(word_df.head(10))

def print_error_analysis(clf, evaluator, size=20):
    evaluator.show_errors(clf, size)


def update_dict(model_dict, model_name, clf, mic, mac, cm, acc, pred):
    model_dict[model_name][CLF] = clf
    model_dict[model_name][MACRO] = mac
    model_dict[model_name][MICRO] = mic
    model_dict[model_name][ACCURACY] = acc
    model_dict[model_name][CONFUSION_MATRIX] = cm
    model_dict[model_name][PREDICTION] = pred
    return model_dict
