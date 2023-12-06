import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def sort_dataset(dataset_df):
    sort_data = dataset_df.sort_values(by='year', ascending=True)
    return sort_data

def split_dataset(dataset_df):
    X=dataset_df.drop(columns="salary", axis=1)
    Y=dataset_df["salary"]*0.001
    X_train=X[:1718]
    X_test=X[1718:]
    y_train=Y[:1718]
    y_test=Y[1718:]
    return X_train,X_test,y_train,y_test

def extract_numerical_cols(dataset_df):
    return dataset_df[['age', 'G','PA','RBI', 'SB', 'CS', 'BB', 'HBP','AB','R', 'H','2B','3B','HR','SO', 'GDP','fly', 'war']]

def train_predict_decision_tree(X_train, Y_train, X_test):
    dtr=DecisionTreeRegressor()
    dtr.fit(X_train,Y_train)
    predited=dtr.predict(X_test)
    return predited

def train_predict_random_forest(X_train, Y_train, X_test):
    rf_reg = RandomForestRegressor()
    rf_reg.fit(X_train,Y_train)
    predited = rf_reg.predict(X_test)
    return predited

def train_predict_svm(X_train, Y_train, X_test):
    svm_pipe = make_pipeline(
        StandardScaler(),
        SVR()
    )
    svm_pipe.fit(X_train, Y_train)
    predited = svm_pipe.predict(X_test)
    return predited


def calculate_RMSE(labels, predictions):
    return np.sqrt(np.mean((predictions-labels)**2))



if __name__ == '__main__':
    # DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

    sorted_df = sort_dataset(data_df)
    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)

    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)

    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)

    print("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
    print("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
    print("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))