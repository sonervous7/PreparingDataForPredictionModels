from classes import CleanDataset
import pandas as pd

path = "content/dengue_features_train.csv"
california_train = pd.read_csv(path)



if __name__ == '__main__':
    print(california_train.head())
    print(california_train[['city', 'week_start_date']].head())

    df = CleanDataset(data=california_train, date_col_name='week_start_date').get_data(impute_missing=True, remove_outliers=True, encoding_method='ordinal')
    print(df.head())

