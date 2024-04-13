from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import defaultdict
import pandas as pd
import pickle
import json
import os

from src.config.load_config import configuration
from src.models.ml import __models__

cfg = configuration('new_aggregator')


def make_features(type_data='train'):
    data = pd.read_csv(os.path.join(cfg['Dataset']['data_path'], f'{type_data}.txt'), sep='\t')
    X = data['title']
    y = data['category']
    
    if type_data == 'train':
        vectorizer = CountVectorizer(stop_words='english')
        features = vectorizer.fit_transform(X)
        os.makedirs(os.path.dirname(cfg['Model']['vectorizer_path']), exist_ok=True)
        pickle.dump(vectorizer, open(cfg['Model']['vectorizer_path'], 'wb'))
    else:
        if os.path.exists(cfg['Model']['vectorizer_path']):
            vectorizer = pickle.load(open(cfg['Model']['vectorizer_path'], 'rb'))
            features = vectorizer.transform(X)
        else:
            raise ("Not exist vecterizer")
    y = y.map(lambda x: cfg['Model']['label_encode'][x])
    return features, y


def evaluate(y_pred, y_test):
    """Evaluate model 
    """
    acc = accuracy_score(y_pred, y_test)
    p = precision_score(y_pred, y_test, average='weighted')
    r = recall_score(y_pred, y_test, average='weighted')
    f1 = f1_score(y_pred, y_test, average='weighted')
    macro_avg = classification_report(y_test, y_pred, target_names=['b', 'e', 'm', 't']) # {'b':0, 'e':1, 'm':2, 't':3}
    print(f"Accuracy score: {acc}")
    print(f"Precision score: {p}")
    print(f"Recall score: {r}")
    print(f"F1 score: {f1}")
    print(f"Macro Average: \n {macro_avg}")
    return f1


def save_model(model, basename, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    pickle.dump(model, open(os.path.join(save_dir, basename + '.pkl'), 'wb'))
    

X_train, y_train = make_features('train')
X_test, y_test = make_features('test')

evals = []
__classifiers__ = defaultdict()
# Training and evaluate
for m_name in __models__:
    hparams = {}
    hparams_pth = os.path.join(cfg['Model']['hyper_param_path'], m_name + '.json')
    if os.path.exists(hparams_pth):
        hparams = json.load(open(hparams_pth, 'r'))
    
    print(f"=== Traing {m_name} model ===")
    print(f"Loading params: {hparams}")
    classifier = __models__[m_name](**hparams).fit(X_train, y_train)
    
    __classifiers__[m_name] = classifier
    
    preds = classifier.predict(X_test)
    print(f"=> Evaluating {m_name} model")
    f1 = evaluate(preds, y_test)
    evals.append({
        'model': m_name,
        'f1_score': f1
    })
    print()

# Select best model
evals = pd.DataFrame(evals)
print("==> Model comparison ...")
print(evals)
max_idx = evals.f1_score.idxmax()
best_model = evals.model[max_idx]

# Save best model
print("==> Saving best model ...")
os.makedirs(cfg['Model']['model_path'], exist_ok=True)
with open(os.path.join(cfg['Model']['model_path'], 'best_model.pkl'), 'wb') as f:
    pickle.dump(__classifiers__[best_model], f)
