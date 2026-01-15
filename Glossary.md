# Gloassary for ML DEVOPS

## Machine Learning Operations
MLops is a set of best practices and methods for an efficient end-to-end development and operation of performant, scalable, reliable, automated and reproducible ML solutions in a real production setting.

## Reproducible Workflow
An orchestrated, tracked and versioned workflow that can be reproduced and inspected.

### Artifact
The product of a pipeline component. It can be a file (an image, a model export, model weights, a text file...) or a directory.

### Component 
One step in a Machine Learning Pipeline. In MLflow, a component is characterized by an environment file (conda.yml if you are using conda), an entry point definition file (MLproject) and one or more scripts or commands to be executed and their supporting code.

### Container
 A technology to package together the entire runtime of a software, i.e., the code itself and all its dependencies and data files. Containers can be spun up quickly, and they run identically across different environments.

### Data Segregation
 The process of splitting the data, for example into train and test sets.

### Environment (runtime)
The environment where a software runs. In mlflow it is described by the conda.yml file (or the equivalent Dockerfile if using Docker).

### Experiment
A tracked and controlled execution of one or more related software components, or an entire pipeline. In W&B the experiment is called group.

### Hyperparameters
The parameters of a model that are set by the user and do not vary during the optimization or fit. They cannot be estimated from the data.

### Job Type
Used by W&B to distinguish different components when organizing the ML pipeline. It is mostly used for the visualization of the pipeline.

### Machine Learning Pipeline
A sequence of one or more components linked together by artifacts, and controlled by hyperparameters and/or configurations. It should be tracked and reproducible.

### Project in Wandb
 All the code, the experiments and the data that are needed to reach a particular goal, for example, a classification of cats vs dogs.

### Run in Wandb
The minimal unit of execution in W&B and in all tracking software. It usually represents the execution of one script or one notebook, but it can sometimes contain more; for example, one script that spawns other scripts.

## Date Exploration and Preparation
### Exploratory Data Analysis (EDA)
An interactive analysis performed at the beginning of the project, to explore the data and learn as much as possible from it. It informs many decisions about the development of a model. For example, we typically discover what kind of pre-processing the data needs before it can be used for training or for inference. It is also important to verify assumptions that have been made about the data and the problem.

### Feature Engineering
The process of creating new features by combining and/or transforming existing features.

### Feature Store
A MLops tool that can store the definition as well as the implementation of features, and serve them for online (real-time) inference with low latency and for offline (batch) inference with high throughput.

## Data Validation
### Alternative Hypothesis
In statistical hypothesis testing, the alternative hypothesis is a statement that contradicts the null hypothesis.

### Deterministic Test
A test that involves a measurement without randomness. For example, measuring the number of columns in a table.

### ETL Pipelines
Extract Transform Load pipelines. They are a classic structure for data pipelines. An ETL pipeline is used to fetch, preprocess and store a dataset.

### Hypothesis Testing
A statistical method to test a null hypothesis against an alternative hypothesis. The main element of HT is a statistical test.

### Non-Deterministic Test
A test that involves a measurement of a random variable, i.e., of a quantity with intrinsic randomness. Examples are the mean or standard deviation from a sample from a population. If you take two different samples, even from the same population, they will not have exactly the same mean and standard deviation. A non-deterministic test uses a statistical test to determine whether an assumption about the data is likely to have been violated.

### Null Hypothesis
In statistical hypothesis testing, the null hypothesis is the assumption that we want to test. For example, in case of the t-test the null hypothesis is that the two samples have the same mean.

### P-Value
The probability of measuring by chance a value for the Test Statistic equal or more extreme than the one observed in the data assuming that the null hypothesis is true.

Statistical Test: An inference method to determine whether the observed data is likely or unlikely to occur if the null hypothesis is true. It typically requires the specification of an alternative hypothesis, so that a Test Statistic (TS) can be formulated and the expected distribution of TS under the null hypothesis can be derived. A statistical test is characterized by a false positive rate alpha (probability of Type I error) and a false negative rate beta (probability of a Type II error). There are many statistical tests available, depending on the null and the alternative hypothesis that we want to probe.

### Test Statistic
A random variable that can be computed from the data. The formula for the TS is specified by the appropriate statistical test that can be chosen once a null hypothesis and an alternative hypothesis have been formulated. For example, to test whether two samples have the same mean (null hypothesis) or a different mean (alternative hypothesis) we can use the t-test. The t-test specifies how to compute the TS appropriate for this case, as well as what is the expected distribution of TS under the null hypothesis.

## Training, Validation and Experiment Tracking
### Experiment Tracking: 
The process of recording all the necessary pieces of information needed to inspect and reproduce a run. We need to track the code and its version, the dependencies and their versions, all the metrics of interests, all the produced artifacts (images, model exports, etc.), as well as the environment where the experiment runs.

### Hyperparameter Optimization: 
The process of varying one or more hyperparameter of a run in order to optimize a metric of interest (for example, Accuracy or Mean Absolute Error).

### Inference Artifact: 
An instance of the Inference Pipeline containing a trained model.

### Inference Pipeline: 
A pipeline constituted of two steps: the pre-processing step and the model. The pre-processing step can be a pipeline on its own, and it manipulates the data and prepares them for the model. The inference pipeline should contain all the pre-processing that needs to happen during model development as well as during production. When the inference pipeline is trained (i.e., it contains a trained model) it can be exported to disk. The export product is called an Inference Artifact.


## ML Scoring and Tracking
Ordinary least squares regression: the oldest ML method, invented in 1805
Receiver Operating Characteristic: a scoring method for classification models invented in 1941
API's: interfaces with computer programs, first described in 1951
Kaggle: a forum for machine learning competitions, launched in 2010
External stakeholders: customers and company leaders who aren't closely involved with your project, but are interested in making sure it's performing well
Data scientists: professionals who work with data, and train models
ML engineers: professionals who optimize, deploy, and monitor ML models
Data engineers: professionals who work with data ingestion and administration

history of regression:
https://www.tandfonline.com/doi/full/10.1080/10691898.2001.11910537#d1e90

model scoring
https://www.sciencedirect.com/science/article/abs/pii/B9780080970868430916?via%3Dihub
