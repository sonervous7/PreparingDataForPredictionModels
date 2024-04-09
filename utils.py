from time import strftime

import pandas as pd


class DateVariable():
    def __init__(self, column: pd.Series):
        self.column = column

    @staticmethod
    def parse_string_date(string_date: str) -> pd.Timestamp:
        converted_str = pd.to_datetime(string_date)
        formated_str = converted_str.strftime('%Y-%m-%d')
        return pd.Timestamp(formated_str)

    def encode_as_number(self) -> pd.Series:
        if not pd.api.types.is_datetime64_any_dtype(self.column):
            self.column = self.column.apply(
                DateVariable.parse_string_date)  # Apply, działa dla każdego elementu danej serii (jakby w pętli)
            # Brak jawnego przekazywania argumentu, ale no apply jest tak zaprojektowany, aby przekazywać automatycznie każdy element serii jako argument

        # Wyodrębnienie składowych daty
        years = self.column.dt.year
        months = self.column.dt.month
        days = self.column.dt.day

        # Kodowanie dat jako liczby
        encoded_dates = years * 365 + (months - 1) * 30 + days

        return encoded_dates


class DataImputation():

    def mode_imputation(self,
                        column: pd.Series):  # Najcześciej wyśtepująca, wypełnianie niekompletnych danych most_common_value z nich
        mode = column.mode()[0]
        return column.fillna(mode)

    def mean_imputation(self, column: pd.Series):  # Średnia, wypełnianie niekompletnych danych średnią z nich
        mean = column.mean()
        return column.fillna(mean)

    def median_imputation(self, column: pd.Series):  # Mediana, wypełnianie niekompletnych danych medianą z nich
        median = column.median()
        return column.fillna(median)
