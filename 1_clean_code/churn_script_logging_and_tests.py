#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import logging
from unittest import mock
from unittest.mock import MagicMock, patch
import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

from churn_library import ChurnLibrary
import pytest
import pandas as pd
import churn_constants as consts

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)

MOCK_DF_CHURN = pd.DataFrame(
    {
        "Churn": [0, 1, 0, 1],
        "Gender": ["M", "M", "F", "F"],
        "Education_Level": ["Uneducated", "Graduate", "High School", "Unknown"],
        "Marital_Status": ["Married", "Single", "Unknown", "Hippie"],
        "Income_Category": ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K"],
        "Card_Category": ["Black", "Blue", "Silver", "White"],
        "Gender_Churn": [0.5, 0.5, 0.5, 0.5],
        "Education_Level_Churn": [0.0, 1.0, 0.0, 1.0],
        "Marital_Status_Churn": [0.0, 1.0, 0.0, 1.0],
        "Income_Category_Churn": [0.0, 1.0, 0.0, 1.0],
        "Card_Category_Churn": [0.0, 1.0, 0.0, 1.0],
        "Customer_Age": [None, None, None, None],
        "Dependent_count": [None, None, None, None],
        "Months_on_book": [None, None, None, None],
        "Total_Relationship_Count": [None, None, None, None],
        "Months_Inactive_12_mon": [None, None, None, None],
        "Contacts_Count_12_mon": [None, None, None, None],
        "Credit_Limit": [None, None, None, None],
        "Total_Revolving_Bal": [None, None, None, None],
        "Avg_Open_To_Buy": [None, None, None, None],
        "Total_Amt_Chng_Q4_Q1": [None, None, None, None],
        "Total_Trans_Amt": [None, None, None, None],
        "Total_Trans_Ct": [None, None, None, None],
        "Total_Ct_Chng_Q4_Q1": [None, None, None, None],
        "Avg_Utilization_Ratio": [0, 1, 0, 1],
    }
)


@pytest.fixture
def test_task():
    return ChurnLibrary()


def test_import(test_task):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df = test_task.import_data("data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


@patch.object(ChurnLibrary, "_calc_stats", return_value="mock_calc")
@patch.object(ChurnLibrary, "_save_eda_plot", return_value="mock_plot")
def test_eda(mock_save_eda_plot, mock_calc_stats, test_task):
    """
    test perform eda function
    """
    mock_df = pd.DataFrame(
        {
            "Churn": [1, 0, 1, 0],
            "Customer_Age": [1, 2, 3, 4],
            "Marital_Status": ["Married", "Single", "Unknown", "Hippie"],
            "Total_Trans_Ct": [1, 2, 3, 4],
        }
    )
    test_task.perform_eda(mock_df)

    assert mock_calc_stats.call_count == 1
    assert mock_save_eda_plot.call_count == 5


def test_encoder_helper(test_task):
    """
    test encoder helper
    """
    mock_df = pd.DataFrame(
        {
            "Churn": [0, 1, 0, 1],
            "Gender": ["M", "M", "F", "F"],
            "Education_Level": ["Uneducated", "Graduate", "High School", "Unknown"],
            "Marital_Status": ["Married", "Single", "Unknown", "Hippie"],
            "Income_Category": ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K"],
            "Card_Category": ["Black", "Blue", "Silver", "White"],
        }
    )

    expected_df = pd.DataFrame(
        {
            "Churn": [0, 1, 0, 1],
            "Gender": ["M", "M", "F", "F"],
            "Education_Level": ["Uneducated", "Graduate", "High School", "Unknown"],
            "Marital_Status": ["Married", "Single", "Unknown", "Hippie"],
            "Income_Category": ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K"],
            "Card_Category": ["Black", "Blue", "Silver", "White"],
            "Gender_Churn": [0.5, 0.5, 0.5, 0.5],
            "Education_Level_Churn": [0.0, 1.0, 0.0, 1.0],
            "Marital_Status_Churn": [0.0, 1.0, 0.0, 1.0],
            "Income_Category_Churn": [0.0, 1.0, 0.0, 1.0],
            "Card_Category_Churn": [0.0, 1.0, 0.0, 1.0],
        }
    )
    result = test_task.encoder_helper(mock_df, consts.COLS_CATEGORIES)
    pd.testing.assert_frame_equal(result, expected_df)


@mock.patch("sklearn.model_selection.GridSearchCV", return_value=MagicMock())
@mock.patch("joblib.dump")
@mock.patch("sklearn.linear_model.LogisticRegression", return_value=MagicMock())
@mock.patch("sklearn.ensemble.RandomForestClassifier", return_value=MagicMock())
@patch.object(ChurnLibrary, "classification_report_image")
@patch.object(ChurnLibrary, "feature_importance_plot")
def test_train_models(
    mock_feature_plot, mock_classification_report, mock_rfc, mock_lrc, mock_joblib, mock_grid_search_cls, test_task
):
    """
    Test the train_models function using MagicMock to simulate `GridSearchCV`.
    """

    mock_grid_search = MagicMock()
    mock_grid_search.fit.return_value = mock_grid_search
    mock_grid_search.best_estimator_.predict.return_value = [0, 1]

    mock_grid_search_cls.return_value = mock_grid_search

    mock_X_train = [[1, 2], [3, 4]]
    mock_X_test = [[11, 12], [13, 14]]
    mock_y_train = [0, 1]
    mock_y_test = [1, 0]

    test_task.train_models(mock_X_train, mock_X_test, mock_y_train, mock_y_test)

    mock_grid_search.fit.assert_called_once_with(mock_X_train, mock_y_train)
    mock_grid_search.best_estimator_.predict.assert_called()

    assert mock_joblib.call_count == 2
    assert mock_classification_report.call_count == 1
    assert mock_feature_plot.call_count == 1


if __name__ == "__main__":
    pass
