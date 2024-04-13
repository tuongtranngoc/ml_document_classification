from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


from sklearn.ensemble import (AdaBoostClassifier, 
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB


# Import regression models
__models__ = {
    'RandomForestClassifier': RandomForestClassifier,
    'GradBoostClassifier': GradientBoostingClassifier,
    'AdaBoostClassifier': AdaBoostClassifier,
    'XGBClassifier': XGBClassifier,
    'MultinomialNB': MultinomialNB,
}
