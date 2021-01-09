import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, f1_score, accuracy_score
from utils import F1, PRECISION, RECALL, ACCURACY


class Evaluator:
    def __init__(self, test_path, test_df=None):
        if test_df is not None:
            self.test_df = test_df
        else:
            self.test_df = pd.read_csv(Path(rf'{test_path}'),
                                       encoding="utf-8",
                                       sep="\t",
                                       header=None,
                                       names=["text", "label"]).reset_index()
        self.test_lbl = test_df.label

    def get_true_labels(self):
        return self.test_df[["index", "label"]]

    def evaluate(self, predicted_labels):
        prec_mic = precision_score(self.test_lbl,
                                   predicted_labels,
                                   average="micro")

        rec_mic = recall_score(self.test_lbl,
                               predicted_labels,
                               average="micro")

        f1_mic = f1_score(self.test_lbl,
                          predicted_labels,
                          average="micro")
        print(f"Micro precision:{prec_mic}, recall:{rec_mic}, f1:{f1_mic}")
        prec_mac = precision_score(self.test_lbl,
                                   predicted_labels,
                                   average="macro")

        rec_mac = recall_score(self.test_lbl,
                               predicted_labels,
                               average="macro")

        f1_mac = f1_score(self.test_lbl,
                          predicted_labels,
                          average="macro")
        print(f"Macro precision:{prec_mac}, recall:{rec_mac}, f1:{f1_mac}")

        cm = confusion_matrix(self.test_lbl,
                              predicted_labels)
        print(cm)
        acc = accuracy_score(self.test_lbl,
                             predicted_labels)

        print(f"Accuracy: {acc}")
        mic = {PRECISION: prec_mic, RECALL: rec_mic, F1: f1_mic}
        mac = {PRECISION: prec_mac, RECALL: rec_mac, F1: f1_mac}
        return mic, mac, cm, acc

    def set_label(self, lbl):
        self.test_lbl = lbl

    def show_errors(self, predicted_labels, name, n=20,model=None):
        print(f'####ERRORS ANALYSIS  for {name}#########')
        pred_df = self.test_df.copy()
        if model is not None:
            model.precdic
        pred_df.loc[:, "pred_label"] = predicted_labels
        wrong_df = pred_df[pred_df.pred_label != pred_df.label]
        print(f"overall, there are {len(wrong_df)} instances with wrong sentiment")
        # num_shown = min(n, len(wrong_df))
        for i in range(0, len(wrong_df)):
            cur_row = wrong_df.iloc[i]
            print(f"{cur_row.text}, true:{cur_row.label}, predicted:{cur_row.pred_label}\n")

    def show_correct(self, predicted_labels, name, n=20, ):
        print(f'#### CORRECT ANALYSIS  for {name}#########')

        pred_df = self.test_df.copy()
        pred_df.loc[:, "pred_label"] = predicted_labels
        right_df = pred_df[pred_df.pred_label == pred_df.label]
        print(f"overall, there are {len(right_df)} instances with right sentiment")
        num_shown = min(n, len(right_df))
        for i in range(0, num_shown):
            cur_row = right_df.iloc[i]
            print(f"{cur_row.text}, true:{cur_row.label}, predicted:{cur_row.pred_label}\n")

