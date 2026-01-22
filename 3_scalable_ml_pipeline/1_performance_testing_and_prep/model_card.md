# What is this
Model cards are a succinct approach for documenting the creation, use, and shortcomings of a model. They should be written such that a non-expert can understand the model card's contents.

There is no one way to write a model card! Suggested sections include:

Model Details such as who made it, type of model, training/hyperparameter details, and links to any additional documentation like a paper reference.
Intended use for the model and the intended users.
Metrics of how the model performs. Include overall performance and also key slices. A figure or two can convey a lot.
Data including the training and validation data. How it was acquired and processed.
Bias inherent either in data or model. This could also be included in the metrics or data section.
Caveats, if there are any.

See also: https://www.tensorflow.org/responsible_ai/model_card_toolkit/guide#model-card-toolkit


# Sample of a model card
## Model Details
Justin C Smith created the model. It is logistic regression using the default hyperparameters in scikit-learn 0.24.2.

## Intended Use
This model should be used to predict the acceptability of a car based off a handful of attributes. The users are prospective car buyers.

## Metrics
The model was evaluated using F1 score. The value is 0.8960.

## Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Car+Evaluation(opens in a new tab)). The target class was modified from four categories down to two: "unacc" and "acc", where "good" and "vgood" were mapped to "acc".

The original data set has 1728 rows, and a 75-25 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Bias
According to Aequitas bias is present at the unsupervised and supervised level. This implies an unfairness in the underlying data and also unfairness in the model. From Aequitas summary plot we see bias is present in only some of the features and is not consistent across metrics.