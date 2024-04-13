from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pickle
import pandas as pd

from src.config.load_config import configuration

cfg = configuration('new_aggregator')


class Predictor:
    def __init__(self) -> None:
        self.model = pickle.load(open(os.path.join(cfg['Model']['model_path'], 'best_model.pkl'), 'rb'))
        self.vectorizer = pickle.load(open(cfg['Model']['vectorizer_path'], 'rb'))
        self.label_decode = {
            v: k for k, v in cfg['Model']['label_encode'].items()
        }
    
    def _predict(self, text):
        if text is None or len(text) == 0:
            return None, None
        # Transform text into features
        transformed_text = self.vectorizer.transform([text])
        pred = self.model.predict(transformed_text)
        prob = self.model.predict_proba(transformed_text)
        prob = prob.max()
        pred_cate = self.label_decode[pred[0]]
        return pred_cate, prob


if __name__ == "__main__":
    valid_data = pd.read_csv(os.path.join(cfg['Dataset']['data_path'], 'valid.txt'), sep='\t')
    samples = valid_data.title.tolist()
    predictor = Predictor()
    res = predictor._predict([samples[0]])  
    print(res)     

        

    