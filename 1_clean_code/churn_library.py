#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Title: Churn Library
Description:
Downlaods given data, performs EDA, trains a model and calculates the feature importance
"""

# import libraries
from abc import ABC
import logging
import os

from sklearn.impute import SimpleImputer
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

import churn_constants as consts


os.environ["QT_QPA_PLATFORM"] = "offscreen"

# configure logging
logger = logging.getLogger()
log_file = os.path.join(consts.LOGS_PATH, "churn_library.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # log to file
        logging.StreamHandler(),  # log to console
    ],
)


class ChurnLibrary(ABC):
    X = pd.DataFrame()

    def run(self):
        """
        Main function that calls all the needed functions

        """
        logger.info("Starting Churn Library")
        df = self.import_data("data/bank_data.csv")
        df["Churn"] = df["Attrition_Flag"].apply(lambda val: 0 if val == "Existing Customer" else 1)
        self.perform_eda(df)
        X_train, X_test, y_train, y_test = self.perform_feature_engineering(df)
        self.train_models(X_train, X_test, y_train, y_test)

    def _calc_stats(self, df):
        """
        Calculates the most important stats for a given dataframe

        input: pandas dataframe
        output: None
        """
        logging.info("Show first rows of data:")
        logging.info(df.head())
        logging.info("Show nulls:")
        logging.info(df.isnull().sum())
        logging.info("DataFrame shape:")
        logging.info(df.shape)
        logging.info("Describe incoming data: ")
        logging.info(df.describe())
        logger.info("Done printing stats of current DF.")

    def _save_eda_plot(self, df, plot_type, column_name=None):
        """
        takes in a plot name

        :param plot_name: name of the plot
        :type plot_name: string
        """
        logger.info(f"Storing {plot_type} plot - {column_name}.")
        plt.figure(figsize=(20, 10))
        if plot_type == "hist":
            df[column_name].hist()
        elif plot_type == "bar":
            df.Marital_Status.value_counts("normalize").plot(kind="bar")
        elif plot_type == "sns_hist":
            # Show distributions of a given column name and add a smooth curve obtained using a kernel density estimate
            sns.histplot(df[column_name], stat="density", kde=True)
        elif plot_type == "heatmap":
            df_heatmap = df.filter(items=consts.COLS_QUANTITIES)
            sns.heatmap(df_heatmap.corr(), annot=False, cmap="Dark2_r", linewidths=2)
        else:
            logger.warning(f"The plot type {plot_type} does not exist.")

        plt.savefig(f"{consts.IMAGES_PATH}eda/{plot_type}_{column_name}.png")
        plt.close()

    def _plot_training_stats(self, algo_name, y_test, y_test_preds, y_train, y_train_preds):
        """
        takes in training scores and plots them and saves them to a given directory as .png

        :param algo_name: Name of the machine learning algorithm used for training.
        :type algo_name: str
        :param y_test: Actual target values for the test dataset
        :type y_test: pandas.Series
        :param y_test_preds: Predicted target values for the test dataset
        :type y_test_preds: pandas.Series
        :param y_train: Actual target values for the training dataset
        :type y_train: pandas.Series
        :param y_train_preds: Predicted target values for the training dataset
        :type y_train_preds: pandas.Series
        """
        # scores for random forest
        plt.rc("figure", figsize=(5, 5))
        plt.text(0.01, 1.25, str(f"{algo_name} Train"), {"fontsize": 10}, fontproperties="monospace")
        plt.text(
            0.01,
            0.05,
            str(classification_report(y_test, y_test_preds)),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        plt.text(0.01, 0.6, str(f"{algo_name}  Test"), {"fontsize": 10}, fontproperties="monospace")
        plt.text(
            0.01,
            0.7,
            str(classification_report(y_train, y_train_preds)),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        plt.axis("off")
        algo_file_name = algo_name.lower().replace(" ", "_")
        plt.savefig(f"{consts.IMAGES_PATH}results/scores_{algo_file_name}.png")
        plt.close()

    def import_data(self, pth):
        """
        returns dataframe from a .csv via a given at pth

        input:
                pth: a path to the csv
        output:
                df: pandas dataframe
        """
        return pd.read_csv(pth)

    def perform_eda(self, df):
        """
        perform exploratory data analysisi on a given dataframe and saves figures to an folder
        input:
                df: pandas dataframe

        output:
                None
        """

        logger.info("Starting EDA")

        self._calc_stats(df)
        self._save_eda_plot(
            df,
            "hist",
            "Churn",
        )
        self._save_eda_plot(
            df,
            "hist",
            "Customer_Age",
        )
        self._save_eda_plot(df, "bar", "Marital_Status")
        self._save_eda_plot(df, "sns_hist", "Total_Trans_Ct")
        self._save_eda_plot(df, "heatmap")

    def encoder_helper(self, df, category_lst, response=None):
        """
        Helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook

        input:
                df: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could be used for naming variables or index y column]

        output:
                df: pandas dataframe with new columns for
        """

        try:
            for category in category_lst:
                df[category] = pd.to_numeric(df[category], errors="coerce")
                category_groups = df.groupby(category)["Churn"].mean()
                df[f"{category}_Churn"] = df[category].map(category_groups)

            return df
        except ValueError as e:
            logger.error(f"Unable to add encoded groups: {e}")

    def perform_feature_engineering(self, df, response=None):
        """
        Calls the train_test_split functionality with selected columns
        input:
                  df: pandas dataframe
                  response: string of response name [optional argument that could be used for naming variables or index y column]

        output:
                  X_train: X training data
                  X_test: X testing data
                  y_train: y training data
                  y_test: y testing data
        """

        logger.info("Starting Feature Engineering")

        df = self.encoder_helper(df, consts.COLS_CATEGORIES)

        y = df["Churn"]
        self.X[consts.COLS_TO_KEEP] = df[consts.COLS_TO_KEEP]
        logger.info("Calling train_test_split")
        return train_test_split(self.X, y, test_size=0.3, random_state=42)

    def classification_report_image(
        self, y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf, X_test
    ):
        """
        stores support scores for the model to logs and folders

        input:
                y_train: training response values
                y_test:  test response values
                y_train_preds_lr: training predictions from logistic regression
                y_train_preds_rf: training predictions from random forest
                y_test_preds_lr: test predictions from logistic regression
                y_test_preds_rf: test predictions from random forest

        output:
            None
        """

        # log scores
        logger.info("random forest results")
        logger.info("test results")
        rf_test_results = classification_report(y_test, y_test_preds_rf)
        logger.info(f"\n{rf_test_results}")
        logger.info("train results")
        rf_train_results = classification_report(y_train, y_train_preds_rf)
        logger.info(f"\n{rf_train_results}")
        logger.info("--------------------------")
        logger.info("logistic regression results")
        logger.info("test results")
        lr_test_results = classification_report(y_test, y_test_preds_lr)
        logger.info(f"\n{lr_test_results}")
        logger.info("train results")
        lr_train_results = classification_report(y_train, y_train_preds_lr)
        logger.info(f"\n{lr_train_results}")

        # loads the models
        rfc_model = joblib.load(f"{consts.MODELS_PATH}rfc_model.pkl")
        lrc = joblib.load(f"{consts.MODELS_PATH}logistic_model.pkl")

        # plots
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        lrc_plot = RocCurveDisplay.from_estimator(lrc, X_test, y_test)
        lrc_plot.plot(ax=ax, alpha=0.8)
        plt.savefig(f"{consts.IMAGES_PATH}results/roc_curve.png")

        explainer = shap.TreeExplainer(rfc_model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        plt.savefig(f"{consts.IMAGES_PATH}results/shap.png")

        self._plot_training_stats("Random Forest", y_test, y_test_preds_rf, y_train, y_train_preds_rf)
        self._plot_training_stats("Logistic Regression", y_test, y_test_preds_lr, y_train, y_train_preds_lr)

    def feature_importance_plot(self, model, X_data, output_pth):
        """
        creates and stores the feature importances in pth
        input:
                model: model object containing feature_importances_
                X_data: pandas dataframe of X values
                output_pth: path to store the figure

        output:
                 None
        """

        # Calculate feature importances
        importances = model.best_estimator_.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [X_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel("Importance")

        # Add bars
        plt.bar(range(X_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(X_data.shape[1]), names, rotation=90)

        plt.savefig(f"{output_pth}results/feature_importance.png")

    def train_models(self, X_train, X_test, y_train, y_test):
        """
        train, store model results: images + scores, and store models
        input:
                  X_train: X training data
                  X_test: X testing data
                  y_train: y training data
                  y_test: y testing data
        output:
                  None
        """

        logger.info("Starting Model Training")

        # initialize models
        rfc = RandomForestClassifier(random_state=42)

        # Use SimpleImputer to replace NaNs with column mean)
        imputer = SimpleImputer(strategy="mean")
        # Fit the imputer on training data & transform both train and test sets
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        try:
            lrc = LogisticRegression(solver="lbfgs", max_iter=3000)
        except ValueError as e:
            logger.error(f"Logistic Regression failed to converge with default, please use a different solver. \n {e}")

        param_grid = {
            "n_estimators": [200, 500],
            "max_features": ["auto", "sqrt"],
            "max_depth": [4, 5, 100],
            "criterion": ["gini", "entropy"],
        }
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

        # train the models
        cv_rfc.fit(X_train, y_train)
        lrc.fit(X_train, y_train)

        # get predictions
        y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
        y_train_preds_lr = lrc.predict(X_train)
        y_test_preds_lr = lrc.predict(X_test)

        # save best model
        joblib.dump(cv_rfc.best_estimator_, f"{consts.MODELS_PATH}/rfc_model.pkl")
        joblib.dump(lrc, f"{consts.MODELS_PATH}logistic_model.pkl")

        # plot the outcome
        self.classification_report_image(
            y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf, X_test
        )
        self.feature_importance_plot(cv_rfc, self.X, consts.IMAGES_PATH)


if __name__ == "__main__":
    """
    Main Function for the churn library script
    """
    try:
        ChurnLibrary().run()
    except Exception as e:
        logger.exception("An error occurred: %s", e)
