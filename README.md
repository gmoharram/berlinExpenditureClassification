# berlinExpenditureClassification

The expenditure category is predicted based on the "Zweck" (cause) sentence.

The data was obtained from:
https://www.berlin.de/sen/finanzen/service/zuwendungsdatenbank/

The steps taken are recorded in 4 jupyter notebooks making use of 4 python modules documented at https://github.com/gmoharram/berlinExpenditureClassification/tree/main/docs/_build/html as html files.

### DataExplorationProcessing

This notebook performs an intitial exploration and processing of the data.
The main results were:
- asserting highly imbalanced classes and setting up stratified sampling for splitting the dataset
- realizing that half of the instances are duplicates and will be discarded
- deciding on the tokenization method for the "Zweck" sentence

### ModelSelectionProcess

This notebook further explores the training dataset, defines and evaluates a baseline model, iterates on multiple models and finally, trains the selected model on the entire training set. The hyperparameters of the final model are informed by the hyperparameter tuning performed in the LSTMHyperparameterOptimization.ipynb notebook.

### LSTMHyperparameterOptimization

This notebook performs hyperparameter optimization on a tensorflow LSTM (units, dropout, recurrent_dropout, activationa and optimizer) in order to inform the training of the final model. The runs are inspected in tensorboard.

### TestPredictionEvaluation

This notebook predicts the labels of the dataset based solely on the "Zweck" sentence with the help of the trained model and the pretrained Word2Vector Embedding stored in https://github.com/gmoharram/berlinExpenditureClassification/tree/main/assets.
The results are subsequently evaluated and model performance metrics are calculated and visualized.

<p align="center">
  <img src="https://github.com/gmoharram/berlinExpenditureClassification/blob/main/assets/img/testEvaluation.png" />
</p>







