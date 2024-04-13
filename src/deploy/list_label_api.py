from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from flask import Flask, jsonify, request
from src.tools.predict import Predictor

predictor = Predictor()

app = Flask(__name__)

@app.route('/list_label', methods=['GET'])
def list_label():
    labels = list(predictor.label_decode.values())
    
    result = {
        'labels': labels
    }
    
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2005, debug=True)