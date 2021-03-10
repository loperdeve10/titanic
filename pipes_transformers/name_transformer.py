import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from pathlib import Path
#
# BASE_PATH = Path("..")
# TRAIN_PATH = BASE_PATH / "train.csv"
# TEST_PATH = BASE_PATH / "test.csv"
# GENDER_SUBMISSION_PATH = BASE_PATH / "gender_submission.csv"
# train_df = pd.read_csv(TRAIN_PATH)
# test_df = pd.read_csv(TEST_PATH)
# gender_sub_df = pd.read_csv(GENDER_SUBMISSION_PATH)
# train_df
# test_df
# all_df = pd.concat([train_df, test_df])
#
#
# def extract_words(name):
#     splitted_words = name.split(" ")
#     splitted_words = [w for w in splitted_words if "." in w]
#     # TODO remove any non a-zA-Z chars
#     return splitted_words
#
#
# def print_social_status(df):
#     all_words = []
#     for name in df.Name:
#         for word in name.split(" "):
#             if "." in word:
#                 all_words.append(word)
#     print(f"possible social status: {set(all_words)}")
#
#
# def print_common_words(df, threshold):
#     threshold = int(df.shape[0] * threshold)
#     print(f"Check {df.shape[0]} records --> threshold={threshold}")
#     all_words = []
#     for name in df.Name:
#         #     print("------")
#         #     print(name)
#         #     print(extract_words(name))
#         for w in name.split(" "):
#             all_words.append(w)
#     all_words_df = pd.DataFrame(all_words)
#     print(all_words_df.value_counts()[all_words_df.value_counts() > threshold])
#
#
# print_social_status(all_df)
# print()


def extract_social_status(name):
    STATUSES = ['Dr.', 'Mrs.', 'Don.', 'Mme.', 'Ms.', 'Lady.', 'Sir.', 'Dona.', 'Jonkheer.', 'Master.', 'Countess.',
                'Rev.', 'Mlle.', 'L.', 'Capt.', 'Col.', 'Mr.', 'Miss.', 'Major.']
    for status in STATUSES:
        if status in name:
            return status
    return None


class NameSocialStatusTransformerV1(BaseEstimator, TransformerMixin):
    """ Extract SocialStatus from name """

    def __init__(self):
        self.columns_name = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if "Name" not in X.columns:
            raise ValueError(f"Name not in {X.columns}")
        X["SocialStatus"] = X.Name.map(lambda name: extract_social_status(name))
        X["SocialStatus"] = X["SocialStatus"].astype('category')

        return X
