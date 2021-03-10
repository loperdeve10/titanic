from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

def find_floor(cabin):
    possible_floor = []
    floors = ["A", "B", "C", "D", "E", "F", "G"]
    for f in floors:
        if f in str(cabin):
            possible_floor.append(f)
    if len(set(possible_floor)) == 1:
        return possible_floor[0]
    elif len(set(possible_floor)) > 1:
        ret = possible_floor[0]
        print(f"Not sure what to do: find_floor ({cabin}), return \"{ret}\"")
        return ret
    else:
        if cabin is not np.nan:
            print(f"Failed: find_floor ({type(cabin), cabin})")
        return None


class CabinFloorTransformerV1(BaseEstimator, TransformerMixin):
    """ Extract Floor from Cabin """

    def __init__(self):
        self.columns_name = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if "Cabin" not in X.columns:
            raise ValueError(f"Cabin not in {X.columns}")
        X["CabinFloor"] = X.Cabin.map(lambda cabin: find_floor(cabin))
        X["CabinFloor"] = X["CabinFloor"].astype('category')

        return X
