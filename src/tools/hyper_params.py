from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import pickle
import pandas as pd

from src.models.ml import __models__
from src.config.load_config import configuration

from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer

cfg = configuration("new_aggregator")


__hyper_params__ = {
    "RandomForestClassifier": {
        'n_estimators':(100,300),
        'max_depth': (3, 40),
    },
    "GradBoostClassifier": {
        'n_estimators':(10,30),
        'learning_rate':(0,1),
        'max_depth': (3, 40),
    },
    "AdaBoostClassifier": {
        'n_estimators':(100,300),
        'learning_rate':(0.2,1),
    },
    "XGBClassifier": {
        'max_depth': (3, 40),
        'gamma': (0.2, 1),
        'learning_rate':(0.2,1),
        'n_estimators':(100,300),
    },
    "MultinomialNB": {
        'alpha': (0.2,1),

    }
}


class HyperTuning:
    def __init__(self) -> None:
        self.X_valid, self.y_valid = self.preprocess_data()
    
    def preprocess_data(self):      
        data = pd.read_csv(os.path.join(cfg['Dataset']['data_path'], 'valid.txt'), sep='\t')
        X = data['title']
        y = data['category']

        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(X)
        y = y.map(lambda x: cfg['Model']['label_encode'][x])
        return X, y
    
    def xgboost_crossval(self, max_depth, gamma, learning_rate, n_estimators):
        params = {
            'max_depth': int(max_depth),
            'gamma': gamma,
            'learning_rate': learning_rate,
            'eval_metric': 'auc',
            'n_estimators': int(n_estimators)
        }
        scores = cross_val_score(__models__['XGBClassifier'](**params,use_label_encoder=False),
                                self.X_valid, self.y_valid, cv=5, scoring="f1_macro").mean()
        return scores.mean()
    
    def gradboost_crossval(self, n_estimators, learning_rate, max_depth):
        params = {
            'learning_rate': learning_rate,
            'max_depth': int(max_depth), 
            'n_estimators': int(n_estimators)
        }
        scores = cross_val_score(__models__['GradBoostClassifier'](**params),
                                self.X_valid, self.y_valid, cv=5, scoring="f1_macro").mean()
        return scores.mean()
    
    def adaboost_crossval(self, n_estimators, learning_rate):
        params = {
            'n_estimators': int(n_estimators),
            'learning_rate': learning_rate,
        }
        scores = cross_val_score(__models__['AdaBoostClassifier'](**params),
                                self.X_valid, self.y_valid, cv=5, scoring="f1_macro").mean()
        return scores.mean()
    
    def randomForest_crossval(self, n_estimators, max_depth):
        params = {
            'n_estimators': int(n_estimators),
            'max_depth': int(max_depth),
        }
        scores = cross_val_score(__models__['RandomForestClassifier'](**params),
                                self.X_valid, self.y_valid, cv=5, scoring="f1_macro").mean()
        return scores.mean()
    
    def multinomialNB_crossval(self, alpha):
        params = {
            'alpha': alpha,
        }
        scores = cross_val_score(__models__['MultinomialNB'](**params),
                                self.X_valid, self.y_valid, cv=5, scoring="f1_macro").mean()
        return scores.mean()
    
    def optimize_xgboost(self):
        model_bo = BayesianOptimization(self.xgboost_crossval, __hyper_params__['XGBClassifier'])
        model_bo.maximize(n_iter=5, init_points=10)
        params = model_bo.max['params']
        params['max_depth']= int(params['max_depth'])
        params['n_estimators']= int(params['n_estimators'])
        os.makedirs(cfg['Model']['hyper_param_path'], exist_ok=True)
        with open(os.path.join(cfg['Model']['hyper_param_path'], 'XGBClassifier.json'), 'w') as f:
            json.dump(params, f, ensure_ascii=False)

    def optimize_gradboost(self):
        model_bo = BayesianOptimization(self.gradboost_crossval, __hyper_params__['GradBoostClassifier'])
        model_bo.maximize(n_iter=5, init_points=10)
        params = model_bo.max['params']
        params['max_depth'] = int(params['max_depth'])
        params['n_estimators']= int(params['n_estimators'])
        os.makedirs(cfg['Model']['hyper_param_path'], exist_ok=True)
        with open(os.path.join(cfg['Model']['hyper_param_path'], 'GradBoostClassifier.json'), 'w') as f:
            json.dump(params, f, ensure_ascii=False)

    def optimize_adaboost(self):
        model_bo = BayesianOptimization(self.adaboost_crossval, __hyper_params__['AdaBoostClassifier'])
        model_bo.maximize(n_iter=5, init_points=10)
        params = model_bo.max['params']
        params['learning_rate'] = params['learning_rate']
        params['n_estimators']= int(params['n_estimators'])
        os.makedirs(cfg['Model']['hyper_param_path'], exist_ok=True)
        with open(os.path.join(cfg['Model']['hyper_param_path'], 'AdaBoostClassifier.json'), 'w') as f:
            json.dump(params, f, ensure_ascii=False)

    def optimize_randomforest(self):
        model_bo = BayesianOptimization(self.randomForest_crossval, __hyper_params__['RandomForestClassifier'])
        model_bo.maximize(n_iter=5, init_points=10)
        params = model_bo.max['params']
        params['max_depth'] = int(params['max_depth'])
        params['n_estimators']= int(params['n_estimators'])
        os.makedirs(cfg['Model']['hyper_param_path'], exist_ok=True)
        with open(os.path.join(cfg['Model']['hyper_param_path'], 'RandomForestClassifier.json'), 'w') as f:
            json.dump(params, f, ensure_ascii=False)

    def optimize_multinomialNB(self):
        model_bo = BayesianOptimization(self.multinomialNB_crossval, __hyper_params__['MultinomialNB'])
        model_bo.maximize(n_iter=5, init_points=10)
        params = model_bo.max['params']
        os.makedirs(cfg['Model']['hyper_param_path'], exist_ok=True)
        with open(os.path.join(cfg['Model']['hyper_param_path'], 'MultinomialNB.json'), 'w') as f:
            json.dump(params, f, ensure_ascii=False)
        

if __name__ == "__main__":
    model_ob = HyperTuning()
    print(f'Optimizing XGBoost model ...')
    model_ob.optimize_xgboost()
    print(f'Optimizing GradientBoost model ...')
    model_ob.optimize_gradboost()
    print(f'Optimizing MultinomialNB model ...')
    model_ob.optimize_multinomialNB()
    print("Optimizing AdaBoost model ...")
    model_ob.optimize_adaboost()
    print("Optimizing Random Forest ...")
    model_ob.optimize_randomforest()