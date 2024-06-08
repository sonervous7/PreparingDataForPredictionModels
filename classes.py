import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from utils import DateVariable


class DataSet:
    def __init__(self, path=None, data=None):
        if data is not None and path is None:
            self.data = data
        elif data is None and path is not None:
            self.path = path
            self.data = pd.read_csv(path)
        elif data is not None and path is not None:
            print(f"Path: {path}, Type of data: {type(data)}")
            raise ValueError("Please provide either path or data, not both")
        else:
            print("Musimy skądś wziąć dane!")
            raise ValueError("Either path or data should be given.")


class CategoricalVariable():
    def __init__(self, column: pd.Series):
        self.column = column

    @staticmethod
    def ordinal_encode(column : pd.Series, show_mapping=False):
        encoder = OrdinalEncoder()
        encoder_fitted = encoder.fit(pd.DataFrame(column))
        encoded_data = encoder.fit_transform(pd.DataFrame(column))
        inverse_transformation = encoder_fitted.inverse_transform(encoded_data)

        if show_mapping:
            values_mapping = {e.tolist()[0]: t.tolist() for t, e in
                              zip(encoded_data, inverse_transformation)}
            return values_mapping

        return pd.Series(encoded_data.flatten(), index=column.index, name=column.name)

    def encode_data(self, method, show_mapping=False) -> pd.DataFrame:
        if method == 'ordinal':
            encoded_df = CategoricalVariable.ordinal_encode(self.column, show_mapping=show_mapping)
        elif method == 'one_hot':
            encoded_df = CategoricalVariable.one_hot_encode(self.column, show_mapping=show_mapping)
        else:
            raise ValueError(f"Encoding method {method} not recognized")
        return encoded_df


class NewCategoricalVariable(CategoricalVariable):
    def __init__(self, column: pd.Series):
        super().__init__(column)

    @staticmethod
    def one_hot_encode(column: pd.Series) -> pd.DataFrame:
        encoder = OneHotEncoder()
        encoded_data = encoder.fit_transform(column.values(-1, 1))

        frame = pd.DataFrame(
            encoded_data,
            columns=encoder.get_feature_names_out([column.name]),
            index=column.index
        )

        return frame


class NumericVariable():
    def __init__(self, column: pd.Series):
        self.column = column

    def detect_outlier_iqr(self) -> list:
        Q1 = np.percentile(self.column,25)
        Q3 = np.percentile(self.column, 75)
        IQR = Q3 - Q1
        upper_limit = Q3 + 1.5 * IQR
        lower_limit = Q1 - 1.5 * IQR

        outliers_mask = (self.column < lower_limit) | (self.column > upper_limit)

        # Uzyskanie indeksów obserwacji odstających
        outliers_indices = self.column[outliers_mask].index.to_list()

        return outliers_indices


class CategoricalData(DataSet):
    """
    A class used to represent a set of categorical variables from some dataset.
    Inherits from the DataSet class.

    Attributes
    ----------
    path : str, optional. Defaults to None.
        The path to the data file.
    data : pandas DataFrame, optional. Defaults to None.
        The data already in DataFrame form.
    max_uniq_vals : int, optional. Defaults to 10.
        The maximum number of unique values a column can have.
        If a column has more unique values than this, it will not be encoded.
        It is useful for avoiding computational overhead when one-hot encoding.

    Methods
    -------
    encode_data(method, show_mapping=False) -> pd.DataFrame
        Encodes the categorical data using the given method.

    """

    def __init__(self, path=None, data=None, max_uniq_vals=10):
        super().__init__(path, data)
        self.cat_data = self.data.select_dtypes(include='object')
        self.unique_values = self.cat_data.nunique()
        self.cols_to_encode = self.unique_values[self.unique_values <= max_uniq_vals].index.tolist()
        self.cat_data = self.data[self.cols_to_encode]

    def encode_data(self, method, show_mapping=False) -> pd.DataFrame:

        encoded_data = {}

        for column in self.cat_data.columns:
            categorical_col = CategoricalVariable(self.cat_data[column])
            encoded_data[column] = categorical_col.encode_data(method, show_mapping)

        df = pd.DataFrame()

        for k, v in encoded_data.items():
            if method == 'ordinal':
                df[k] = v
            elif method == 'one_hot':
                df = pd.concat([df, v], axis=1)
            else:
                raise ValueError(f"Encoding method {method} not recognized.")

        return df


class NumericData(DataSet):
    """
    A class used to represent numeric data from some dataset.
    Inherits from the DataSet class (which can consist of both numeric and categorical data).

    Attributes
    ----------
    path : str, optional. Defaults to None.
        The path to the data file.
    data : pandas DataFrame, optional. Defaults to None.
        The data already in DataFrame form.

    Methods
    -------
    detect_outliers(method='iqr', by_column=False) -> list or dict
        Detects outliers using the given method.
        Defaults to the interquartile range method.
    """

    def __init__(self, path=None, data=None):

        super().__init__(path, data)
        self.num_data = self.data.select_dtypes(include='number')

    def detect_outliers(self, method='iqr', by_column=False):

        """
        Detects outliers using the given method.
        By default uses interquartile range method.
        Under the hood, it applies NumericVariable.detect_outlier_iqr() to each numeric column,
        and returns combined data as list / dictionary.

        Parameters
        ----------
        method : str, optional. Defaults to 'iqr'.
            The method to use for outlier detection.

        by_column : bool, optional. Defaults to False.
            If True, returns a dictionary of outliers for each column.
            If False, returns a list of indices of the outliers in the dataset (which is a set of all columns).

        Returns
        -------
        list or dict
            A list of indices of the outliers in the dataset (if by_column is False).
            A dictionary of outliers for each column (if by_column is True).
        """

        outliers = {}

        for c in self.num_data.columns:

            numeric_col = NumericVariable(self.data[c])

            if method == 'iqr':
                indices_outliers_iterab = numeric_col.detect_outlier_iqr()

                if indices_outliers_iterab != []:
                    outliers[c] = indices_outliers_iterab


            elif method == 'z_score':
                pass
            else:
                raise ValueError(f"Outlier detection method {method} not recognized.")

        if by_column:
            return outliers

        outlier_indices = []

        for v in outliers.values():
            outlier_indices += v

        return outlier_indices


class PreparingDataset(CategoricalData, NumericData):
    """
    A class used to describe how to prepare data for predictive modelling.
    Inherits from both CategoricalData and NumericData classes.

    Attributes
    ----------
    path : str, optional. Defaults to None.
        The path to the data file.
    data : pandas DataFrame, optional. Defaults to None.
        The data already in DataFrame form.
    date_col_name : str, optional. Defaults to None.
        The name of the column that contains date data.

    Methods
    -------
    prepare_categoricl_data(method='one_hot', impute_missing=False) -> pd.DataFrame
        Prepares the categorical data for predictive modelling, meaning it
        fills in the missing values with the mode (most common value)
        and encodes values such as 'green' as numbers."""

    def __init__(self, path=None, data=None, date_col_name=None):

        CategoricalData.__init__(self, path=path, data=data)
        NumericData.__init__(self, path=path, data=data)

        if date_col_name is not None:
            self.date_data = self.data[date_col_name]

    def prepare_categoricl_data(self, method='one_hot', impute_missing=False):

        if impute_missing:
            for c in self.cat_data.columns:
                most_common = self.cat_data[c].mode()[0]
                self.cat_data[c] = self.cat_data[c].fillna(most_common)

        return self.encode_data(method=method)

    def prepare_numeric_data(self, method='iqr', remove_outliers=False, impute_missing=False):

        outliers = self.detect_outliers(method)
        outliers_by_column = self.detect_outliers(method, by_column=True)

        if impute_missing:
            for c in self.num_data.columns:
                if c in outliers_by_column.keys():
                    median = self.num_data[c].median()
                    self.num_data[c] = self.num_data[c].fillna(median)
                else:
                    mean = self.num_data[c].mean()
                    self.num_data[c] = self.num_data[c].fillna(mean)

        if remove_outliers:
            indices_keep = [i for i in self.num_data.index if i not in outliers]
            self.num_data = self.num_data.iloc[indices_keep]

            return self.num_data

        else:

            return self.num_data

    def prepare_date_data(self):

        return DateVariable(self.date_data).encode_as_number()


class CleanDataset(PreparingDataset):
    """
    A class used to represent a clean dataset which is ready for predictive modelling.

    Attributes
    ----------
    path : str, optional. Defaults to None.
        The path to the data file.
    data : pandas DataFrame, optional. Defaults to None.
        The data already in DataFrame form.
    date_col_name : str, optional. Defaults to None.
        The name of the column that contains date data.

    Methods
    -------
    get_data(encoding_method='one_hot', outlier_method='iqr', remove_outliers=True, impute_missing=False) -> pd.DataFrame
        Returns the clean dataset ready for predictive modelling.

    """

    def __init__(self, path=None, data=None, date_col_name=None):
        super().__init__(path=path, data=data, date_col_name=date_col_name)
        self.date_col_name = date_col_name

    def get_data(self, encoding_method='one_hot', outlier_method='iqr', remove_outliers=True, impute_missing=False):
        categorical = self.prepare_categoricl_data(method=encoding_method, impute_missing=impute_missing)
        numeric = self.prepare_numeric_data(method=outlier_method, remove_outliers=remove_outliers,
                                            impute_missing=impute_missing)

        data_parts = [categorical, numeric]

        if self.date_col_name is not None:
            date_calendar = self.prepare_date_data()
            data_parts.append(date_calendar)

        return pd.concat(data_parts, axis=1)
