from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification
import numpy as np
import tensorflow as tf

def prediction(model,predict_input):
    
    tf_output = model.predict(predict_input)[0]
    tf_prediction = tf.nn.softmax(tf_output, axis=1)
    label = tf.argmax(tf_prediction, axis=1)
    label = label.numpy()
    return int(label[0])

def predict_req(sentence):
    
    loaded_model = TFDistilBertForSequenceClassification.from_pretrained('./sentiment')
    tokenizer = DistilBertTokenizerFast.from_pretrained('./sentimentTokenizer')
    predict_input = tokenizer.encode(sentence,
                                 truncation=True,
                                 padding=True,
                                 return_tensors="tf")
    return prediction(loaded_model,predict_input)
