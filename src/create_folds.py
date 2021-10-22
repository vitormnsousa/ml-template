#imports
import pandas as pd
from sklearn import model_selection

if __name__ == '__main__':

    #training data
    df = pd.read_csv('input/train.csv')

    #create a new column called kfold
    df['kfold'] = -1

    #randomize rows
    df = df.sample(frac=1).reset_index(drop =True)

    #fecth target column
    #y = df.target.values

    #initialize the kfold class
    kf = model_selection.KFold(n_splits=5)

    #fill the column
    for fold, (trn_,val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold

    #save new csv with kfold columns
    df.to_csv('input/train_folds.csv', index=False)
