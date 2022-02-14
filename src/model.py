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

from collections import Counter

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_addons as tfa

import sklearn.tree as tree

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import f1_score, multilabel_confusion_matrix, recall_score

def top_words(processed_data, k, category = None, return_overlap_ratio = False):
    """Takes top k words for every class to include in bag-of-words underlying set
    
    Parameters
    ----------
    processed_data: DataFrame,
                    Processed training data with processed 'Zweck' column as list of words.
    k: int,
       Number of top words to include for each class.
    category: str (default: None),
              Category to investigate, defaults to combining the top words of each category.
    return_overlap_ratio: bool (default: False),
                          Returns ratio of words in top k of multiple classes if True.
                          
    
    Returns
    -------
    bag_set: set,
             Underlying set to use for bag-of-words sentence embedding.
    overlap_ratio: float (optional),
                   Ratio of words in top k of multiple classes.
                
    """
    if category == None:
        categories = processed_data['Politikbereich'].unique()
    else:
        categories = [category]
     
    top_words = []
    for cat in categories :
        cat_words = processed_data[processed_data['Politikbereich'] == cat]['Zweck'].sum()
        cat_words_dist = Counter(cat_words)
        cat_words_dist = pd.DataFrame(sorted(cat_words_dist.items(), key=lambda x: x[1], reverse=True), columns = ['Wort','Anzahl'])
        top_words.extend(cat_words_dist['Wort'][:k])
    top_words = set(top_words)
    
    if return_overlap_ratio:
        overlap_ratio = 1 - len(top_words)/(len(categories)*k)
        
        return top_words, overlap_ratio
    
    else:
        
        return top_words
    

def create_w2v_lookup(tokenized_data, window_size = 2, negative_samples = 5, embedding_dim = 128):
    """Take a corpus of documents and create a word2vector lookup table.
    
    Parameters
    ----------
    tokenized_data: DataFrame,
                    Processed training data with processed 'Zweck' column.
    window_size: int (default: 2),
                 Number of words to consider similar to the left and right of the target word.
    negative_samples: int (default: 5),
                 Number of negative context words to sample.
    
    embedding_dim: int (default: 128),
                   Dimension of the real space the words will be embedded in.
    
    Returns
    -------
    lookup_table: DataFrame,
                  Table to look up the word2vector mapping of the corpus vocabulary.
                  
    """
    
    #don't modify input data
    data = tokenized_data.copy()
    
    #Create index to word mapping
    vocab = top_words(data, None)
    vocab = list(vocab)
    vocab.insert(0, '<pad>')
    indices = list(range(0, len(vocab)))
    vocab2index = dict(zip(vocab, indices))
    
    vocab_size = len(vocab)
    
    
    #Create sentence vector of word indices and add padding for uniform sentence lengths
    data['vectorized_sentence'] = data['Zweck'].apply(lambda row: [vocab2index[x] for x in row])
    max_length = data['vectorized_sentence'].apply(lambda sentence: len(sentence)).max()
    data['padded_sentence'] = data['vectorized_sentence'].apply(lambda sentence: sentence + (max_length -len(sentence))*[0])
    
    
    #function to generate negative sampling skip gram training data
    def generate_targets_contexts_labels(skip_gram_pairs, window_size, num_ns, vocab_size, random_state = 0):

        targets, contexts, labels = [], [], []
        for target_word, context_word in skip_gram_pairs:
            context_class = tf.expand_dims( tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates = tf.random.log_uniform_candidate_sampler(true_classes=context_class,
                                                                                 num_true=1,
                                                                                 num_sampled=num_ns,
                                                                                 unique=True,
                                                                                 range_max=vocab_size,
                                                                                 seed=random_state,
                                                                                 name="negative_sampling")[0]

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)
            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0]*num_ns, dtype="int64")
    
            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

        return targets, contexts, labels

    #Create positive skip gram training data pairs
    
    # Build the sampling table for vocab with `vocab_size` words
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
    skip_gram_pairs = data['padded_sentence'].apply(lambda sentence: tf.keras.preprocessing.sequence.skipgrams(sentence,vocabulary_size=vocab_size, sampling_table=sampling_table,
 window_size=window_size, negative_samples=0)[0])
    skip_gram_pairs = [pair for sublist in skip_gram_pairs.tolist() for pair in sublist]
    
    #Create training data with negative samples
    targets, contexts, labels = generate_targets_contexts_labels(skip_gram_pairs, window_size = window_size, num_ns = negative_samples, vocab_size = vocab_size, random_state = 0)
    
    targets = np.array(targets)
    contexts = np.array(contexts)[:,:,0]
    labels = np.array(labels)
    
    BATCH_SIZE = 1024
    BUFFER_SIZE = targets.size
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    
    class Word2Vec(tf.keras.Model):
        def __init__(self, vocab_size, num_ns, embedding_dim):
            super(Word2Vec, self).__init__()
            self.target_embedding = tf.keras.layers.Embedding(vocab_size,
                                  embedding_dim,
                                  input_length=1,
                                  name="w2v_embedding")
            self.context_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,input_length=num_ns+1)

        def call(self, pair):
            target, context = pair
            
            if len(target.shape) == 2:
                target = tf.squeeze(target, axis=1)
            word_emb = self.target_embedding(target)
            context_emb = self.context_embedding(context)
            dots = tf.einsum('be,bce->bc', word_emb, context_emb)
            return dots
    
    word2vec = Word2Vec(vocab_size, negative_samples,  embedding_dim)
    word2vec.compile(optimizer='adam',
             loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
    
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])
    
    weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
    lookup_table = pd.DataFrame(vocab, columns = ['word'])
    lookup_table['vector'] = weights.tolist()
    
    return lookup_table

def naive_bayes_bow_model(data, tokenized = False, test = True):
    """Classify sentence based on naive bayes model on bag-of-words tokenized sentence represenation.
    
    Parameters
    ----------
    data: DataFrame,
          Data with 'Zweck' column.
    tokenized: bool (default: False),
               If True, 'Zweck' column is assumed to have already been tokenized.
    test: bool (default: True),
               If True, 20% of the data will be used for testing.
               
    Returns
    -------
    nb: sklearn.naive_bayes.MultinomialNB,
        trained naive bayes classification model.
    performance: DataFrame (optional: if "test"),
                 dataframe with classifier performance metrics.
                 
    """
    
    nb_model_data = data.copy()
    
    if not tokenized:
        nb_model_data = process.tokenize(nb_model_data)
    
    nb_model_data['Zweck'] = nb_model_data['Zweck'].apply(lambda x: " ".join(x))
    
    if test:
        train_set, test_set = process.stratified_train_test_split(nb_model_data, test_ratio = .2)
    
        X_train, X_test = np.array(train_set['Zweck'].tolist()), np.array(test_set['Zweck'].tolist()) 
        y_train, y_test = train_set['Politikbereich'], test_set['Politikbereich'] 
        
    else:
        X_train = np.array(nb_model_data['Zweck'].tolist())
        y_train = nb_model_data['Politikbereich']
        
    nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
    nb.fit(X_train, y_train)
        
    if not test:
        return nb
    
    else:
        
        y_pred = nb.predict(X_test)
        
        performance = pd.DataFrame(f1_score(y_test, y_pred, average = None, labels = y_test.unique()), index = y_test.unique(), columns =  ['f1_score'])
        performance['confusion_matrix_TN_FP_FN_TP'] = multilabel_confusion_matrix(y_test, y_pred, labels = y_test.unique()).tolist()
        performance['recall_score'] = recall_score(y_test, y_pred, average = None, labels = y_test.unique())
        
        return nb, performance
        
        
        
    
        

def mean_word_model(data, lookup_table,tokenized = False, transformed = False, test = True, dim_reduction = False):
    """Classify sentence based on mean word embedding with possible dimensionality reduction.
    
    Parameters
    ----------
    data: DataFrame,
          Data with 'Zweck' column.
    lookup_table: DataFrame,
                  Table to look up the word2vector mapping of the corpus vocabulary.
    tokenized: bool (default: False),
               If True, 'Zweck' column is assumed to have already been tokenized.
    transformed: bool (default: False),
                 If True, 'Zweck' column is assumed to have already beeen vectorized.
    test: bool (default: True),
               If True, 20% of the data will be used for testing.
    dim_reduction: bool (default: False),
                   If True, dimensionality reduction on word embeddings is performed before classification.
    
    
    Returns
    -------
    clf: sklearn.tree,
         trained decision tree classification model.
    performance: DataFrame (optional: if "test"),
                 dataframe with classifier performance metrics.
                 
    """
    
    
    mean_model_data = process.create_mean_model_data(data, lookup_table, tokenized = tokenized, transformed = transformed)
    
    if test:
        train_set, test_set = process.stratified_train_test_split(mean_model_data, test_ratio = .2)
    
        X_train, X_test = np.array(train_set['mean_word_vector'].tolist()), np.array(test_set['mean_word_vector'].tolist()) 
        y_train, y_test = train_set['Politikbereich'], test_set['Politikbereich']
            
        
        #dimensionality reduction while keeping around 90% of the "energy"
        if dim_reduction:
            d = X_train.shape[1] #original dimension
            S = np.cov(X_train.T)
            L,U = np.linalg.eig(S)
            energy = 1
            while energy > .9 and d > 0:
                d -= 1
                ind = np.argpartition(L, -d)[-d:]
                energy = sum(L[ind])/sum(L)
            X_train = np.matmul(X_train, U[:, ind])
            X_test = np.matmul(X_test, U[:, ind])
    
        #Train and test classifier
        clf = tree.DecisionTreeClassifier(random_state=0, class_weight = 'balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        performance = pd.DataFrame(f1_score(y_test, y_pred, average = None, labels = y_test.unique()), index = y_test.unique(), columns =  ['f1_score'])
        performance['confusion_matrix_TN_FP_FN_TP'] = multilabel_confusion_matrix(y_test, y_pred, labels = y_test.unique()).tolist()
        performance['recall_score'] = recall_score(y_test, y_pred, average = None, labels = y_test.unique())
        
        return clf, performance
    
    else:
        
        X= np.array(mean_model_data['mean_word_vector'].tolist())
        y= mean_model_data['Politikbereich']
        
        #dimensionality reduction while keeping around 90% of the "energy"
        if dim_reduction:
            d = X.shape[1] #original dimension
            S = np.cov(X.T)
            L,U = np.linalg.eig(S)
            energy = 1
            while energy > .9 and d > 0:
                d -= 1
                ind = np.argpartition(L, -d)[-d:]
                energy = sum(L[ind])/sum(L)
            X = np.matmul(X, U[:, ind])
        
        clf = tree.DecisionTreeClassifier(random_state=0, class_weight = 'balanced')
        clf.fit(X, y)
        
        return clf
    
def plot_performance(performance, score):
    """ Plot a performance score of the classifier by class and inlcude mean line.
    
    Parameters
    ----------
    performance: DataFrame,
                 Dataframe with classes as indices and performance evaluations as columns.
    score: str,
           Performance score column to plot.
    
    Returns
    -------
    performance_plot: AxesSubplot,
                      Plot of performance scores by class.
                      
    """
    
    ax = performance.sort_values(by = score, ascending=False).plot.bar(y = score, title = 'Classifier {} by Class'.format(score), figsize = (12,4), fontsize = 14)
    mean_score = performance[score].mean()
    ax.annotate('Mean Score: {:.2f}'.format(mean_score), xy=(20, mean_score + 0.03))
    return ax.axhline(y=mean_score, color='k', linestyle='--', lw=2)

def lstm_model(data, lookup_table, num_units = 200, dropout = 0.3, rec_dropout = 0.0, activation = 'relu', optimizer = 'adam', tokenized = False, transformed = False, test = True):
    """Classify sentence with lstm model on word embeddings.
    
    Parameters
    ----------
    data: DataFrame,
          Data with 'Zweck' column.
    lookup_table: DataFrame,
                  Table to look up the word2vector mapping of the corpus vocabulary.
    num_units: int (default: 200), 
               Hyperparameter: Dimensionality of LSTM output space.
    dropout: float (default: 0.3),
             Hyperparameter: Fraction of the units to drop for the linear transformation of the inputs.
    rec_dropout: float (default: 0),
                 Hyperparameter: Fraction of the units to drop for the linear transformation of the recurrent state.
    activiation: str (default: 'relu'),
                 Hyperparameter: Activation function to use.
    optimizer: str (default: 'adam'),
                 Hyperparameter: Name of optimizer to use.
    tokenized: bool (default: False),
               If True, 'Zweck' column is assumed to have already been tokenized.
    transformed: bool (default: False),
                 If True, 'Zweck' column is assumed to have already beeen vectorized.
    test: bool (default: True),
          If True, 20% of the data will be used for testing.
    
    
    Returns
    -------
    model: tf.keras.Model,
           trained lstm model which takes tensors in the shape (n, padded_sentence_length, embedding_dimension) as input
           and produces one-hot encoded labels as output.
    performance: DataFrame (optional: if "test"),
                 dataframe with model performance metrics.

    """
    
    lstm_model_data = process.create_lstm_model_data(data, lookup_table, preembed = False, tokenized = tokenized, transformed = transformed)
    
    if test:
        train_set, test_set = process.stratified_train_test_split(lstm_model_data, test_ratio = .2)
        
        input_tensor_test = tf.constant(test_set['padded_sentence'].tolist())
        labels_test = tf.constant(pd.get_dummies(test_set['Politikbereich']))
        BATCH_SIZE = 1000
        BUFFER_SIZE = labels_test.shape[0]
        dataset_test = tf.data.Dataset.from_tensor_slices((input_tensor_test, labels_test))
        dataset_test = dataset_test.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        
    else:
        
        train_set = lstm_model_data
        
    #prepare input for lstm model
    input_tensor = tf.constant(train_set['padded_sentence'].tolist())
    labels = tf.constant(pd.get_dummies(train_set['Politikbereich']))
    BATCH_SIZE = 1000
    BUFFER_SIZE = labels.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        
    #Define model
    weights = np.array(lookup_table['vector'].tolist())
    num_classes = train_set['Politikbereich'].unique().size

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=weights.shape[0],
              output_dim=weights.shape[1],
              embeddings_initializer=tf.keras.initializers.Constant(weights),
              trainable=False,
              mask_zero=True))
    model.add(tf.keras.layers.Masking(mask_value=0.0))
    model.add(tf.keras.layers.LSTM(num_units, return_sequences=False, 
               dropout=dropout, recurrent_dropout=rec_dropout))
    model.add(tf.keras.layers.Dense(num_classes, activation=activation)) 
    
    if activation == 'softmax':
        model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=[tf.keras.metrics.CategoricalAccuracy() , tfa.metrics.F1Score(num_classes = num_classes, average = 'macro')])
    else:
        model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.CategoricalAccuracy(), tfa.metrics.F1Score(num_classes = num_classes, average = 'macro')])
        
    
    #Fit model
    
    if test:
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)]
        model.fit(dataset, validation_data=dataset_test, callbacks = callbacks, epochs=1000)
    else:
        model.fit(dataset, epochs = 60)
        
    if test:
        prediction = model.predict(input_tensor_test, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)
        classes = pd.get_dummies(train_set['Politikbereich']).columns
        y_test = test_set['Politikbereich']
        y_pred = [classes[x] for x in prediction.argmax(axis=1)]
        performance = pd.DataFrame(f1_score(y_test, y_pred, average = None, labels = classes), index = classes, columns =  ['f1_score'])
        performance['confusion_matrix_TN_FP_FN_TP'] = multilabel_confusion_matrix(y_test, y_pred, labels = classes).tolist()
        performance['recall_score'] = recall_score(y_test, y_pred, average = None, labels = classes)
    
        return model, classes, performance
    
    else:
        
        classes = pd.get_dummies(train_set['Politikbereich']).columns
        return model, classes
    