import csv
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from pipes_transformers.baseline import BaselineV1Transformer, BaselineV2Transformer, BaselineV3Transformer, \
    BaselineV4Transformer
from titanic.pipes_transformers.cabin_transformer import CabinFloorTransformerV1
from titanic.pipes_transformers.name_transformer import NameSocialStatusTransformerV1


def load_data(base_dir=Path(".")):
    df_train = pd.read_csv(base_dir / "train.csv")
    df_test = pd.read_csv(base_dir / "test.csv")
    df_anno_example = pd.read_csv(base_dir / "gender_submission.csv")
    return df_train, df_test, df_anno_example


def split_to_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


def evaluate_model(X, pipe, y_true):
    LOG_FILE_PATH = "experiments_results.csv"
    y_predicted = pipe.predict(X)
    print("f1_score macro", f1_score(y_true, y_predicted, average='binary'))
    print("accuracy_score", accuracy_score(y_true, y_predicted))
    ordered_fieldnames = OrderedDict([('pipe_name', None), ('f1_score', None), ('accuracy_score', None)])
    with open(LOG_FILE_PATH, "a") as f:
        dw = csv.DictWriter(f, fieldnames=ordered_fieldnames)
        if f.tell() == 0:
            dw.writeheader()
        dw.writerow({
            'pipe_name': get_pipe_name(pipe),
            'f1_score': f1_score(y_true, y_predicted, average='binary'),
            'accuracy_score': accuracy_score(y_true, y_predicted)
        })


def create_submission_file(X, pipe):
    SUB_DIR = Path("submission_files")
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    y_submission = pipe.predict(X)
    df = pd.DataFrame({"PassengerId": X['PassengerId'], "Survived": y_submission}, columns=['PassengerId', 'Survived'])
    df.to_csv(SUB_DIR / f"{get_pipe_name(pipe)}.csv", index=False)


def get_pipe_name(pipe):
    return "--".join(pipe.steps[i][0] for i in range(len(pipe.steps)))


def get_pipe_docs(pipe):
    print(pipe)
    docs = []
    for i in range(len(pipe.steps) - 1):
        doc_str = pipe.steps[i][1].__doc__ or ""
        if len(doc_str) < 100:
            docs.append(doc_str)
    return ",\n".join(docs)


def print_feature_importance(pipe, plot=True):
    clf = pipe[-1][-1]
    feature_importance = np.array(clf.feature_importances_)
    feature_names = np.array(pipe[-2].columns_name)
    fi_df = pd.DataFrame({'feature_names': feature_names, 'feature_importance': feature_importance})
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    print(f"All feature_names: {fi_df['feature_names'].to_list()}")
    print(f"Best feature_names: {fi_df[fi_df.feature_importance > 0.05]['feature_names'].to_list()}")

    if plot:
        fi_df = fi_df[fi_df.feature_importance > 0.05]
        plt.figure(figsize=(5, 4))
        sns.barplot(y=fi_df['feature_importance'], x=fi_df['feature_names'])
        plt.title(f"Feature Importance: {get_pipe_name(pipe)}")
        plt.xlabel('Feature Names')
        plt.ylabel('Feature Importance')
        plt.show()


def run_full_experiment(pipe, feature_importance=False):
    print("\n")
    print("-" * 60)
    print(f"run experiment {get_pipe_name(pipe)}:\n{get_pipe_docs(pipe)}")

    df_train, df_test, df_anno_example = load_data()
    X_train, X_test, y_train, y_test = split_to_train_test(df_train.copy(), df_train.copy().Survived)

    pipe.fit(X_train, y_train)
    evaluate_model(X_test, pipe, y_test)

    print_feature_importance(pipe, feature_importance)

    create_submission_file(df_test, pipe)


# kaggle competitions submit -c titanic -f %d_baseline.csv -m "%d_baseline"
if __name__ == "__main__":
    run_full_experiment(
        Pipeline([('baseline_v1', BaselineV1Transformer()), ('RF', RandomForestClassifier(random_state=42))]))
    run_full_experiment(
        Pipeline([('baseline_v2', BaselineV2Transformer()), ('RF', RandomForestClassifier(random_state=42))]))
    run_full_experiment(
        Pipeline([('baseline_v3', BaselineV3Transformer()), ('RF', RandomForestClassifier(random_state=42))]))
    run_full_experiment(
        Pipeline([('baseline_v4', BaselineV4Transformer()), ('RF', RandomForestClassifier(random_state=42))]))

    run_full_experiment(
        Pipeline([
            ('name_v1', NameSocialStatusTransformerV1()),
            ('baseline_v4', BaselineV4Transformer()),
            ('RF', RandomForestClassifier(random_state=42))
        ]))

    run_full_experiment(
        Pipeline([
            ('name_v1', NameSocialStatusTransformerV1()),
            ('Cabin_v1', CabinFloorTransformerV1()),
            ('baseline_v4', BaselineV4Transformer()),
            ('RF', RandomForestClassifier(random_state=42))
        ]), feature_importance=True)
