#import os
#import sys
#import inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0, parentdir) 


### replace path with local path or use the above if not using jupyter notebooks

import sys
sys.path.insert(0, "C:\\Users\\gmoha\\OneDrive\\Desktop\\LearningJourney\\Projects\\berlinExpenditureClassification\\src")
import process

import pickle
from keras.models import load_model

import tensorflow as tf
import tensorflow_addons as tfa

def predict_label(data, lookup_path, model_path, model_classes_path, tokenized = False, transformed = False):
    """Classify sentence based on mean word embedding with possible dimensionality reduction.
    
    Parameters
    ----------
    data: DataFrame,
          Data with 'Zweck' column.
    lookup_path: str,
                 Path to Lookup Table for the word2vector mappings of the corpus vocabulary.
    model_path: str,
                 Path to trained prediction model.
    model_classes_path: str,
                        Path to trained model classes to convert probabilities to label.
    tokenized: bool (default: False),
               If True, 'Zweck' column is assumed to have already been tokenized.
    transformed: bool (default: False),
                 If True, 'Zweck' column is assumed to have already beeen vectorized.
                 
    Returns
    -------
    y_predict: Series,
               Labels predicted for input data.
    
    """
    
    #import pretrained objects
    lookup_table = pickle.load(open(lookup_path, "rb" ))
    model = load_model(model_path)
    model_classes = pickle.load(open(model_classes_path, "rb"))
    
    #prepare data
    lstm_model_data = process.create_lstm_model_data(data, lookup_table, preembed = False, tokenized = tokenized, transformed = transformed)
    input_tensor = tf.constant(lstm_model_data['padded_sentence'].tolist())
    
    #make prediction
    
    prediction = model.predict(input_tensor, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)
    y_pred = [model_classes[x] for x in prediction.argmax(axis=1)]
    
    return y_pred