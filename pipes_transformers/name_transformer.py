from sklearn.base import BaseEstimator, TransformerMixin


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
