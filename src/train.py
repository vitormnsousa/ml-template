import os
import argparse

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

import config
import model_dispatcher

def run(fold, model):
    #read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    #to replace by feature engineering
    df = df.drop(['Name', 'Sex','Embarked','Ticket', 'Fare', 'Cabin'], axis=1)
    df.dropna(inplace=True)

    #training data
    df_train = df[df.kfold != fold].reset_index(drop=True)

    #validation data
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    #drop label column from df and convert to array
    x_train= df_train.drop('Survived', axis=1).values
    x_valid= df_valid.drop('Survived', axis=1).values

    #target is label column
    y_train = df_train.Survived.values
    y_valid = df_valid.Survived.values

    #simple Decision Tree model
    clf = model_dispatcher.models[model]

    #fit the model using
    clf.fit(x_train, y_train)

    #create predictions
    preds = clf.predict(x_valid)

    #calculate and print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f'Fold={fold}, Accuracy={accuracy}')

    #save the model
    joblib.dump(
        clf, 
        os.path.join(config.MODEL_OUTPUT, f'df_{fold}.bin')
    )

if __name__ == '__main__':
    #initialize ArgumentParser class
    parser = argparse.ArgumentParser()

    #add the arguments needed
    parser.add_argument(
        '--fold', 
        type=int)
    parser.add_argument(
        '--model', 
        type=str)

    #read the arguments from the comand Lines
    args = parser.parse_args()

    #run the fold specified by command line arguments
    run(
        fold=args.fold,
        model=args.model
    )