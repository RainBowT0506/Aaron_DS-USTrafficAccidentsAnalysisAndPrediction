# Import packages

## Basic data processing
import numpy as np
import pandas as pd

## Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import folium
import folium.plugins

## Modelling
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score , roc_auc_score, balanced_accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost.sklearn import XGBClassifier, XGBRegressor

## Model Explanatory
import graphviz
from pdpbox import pdp , info_plots

## Settings
import warnings

warnings.filterwarnings('ignore')

# 讀取資料集並將其載入到 pandas 資料框中
# read & load the dataset into pandas dataframe
data_df = pd.read_csv('small_data.csv', encoding='ISO-8859-1')
pd.set_option('display.max_columns', 500) # Able to display more columns.
data_df.info()

# Have a look
data_df.head(5)


# 假設 data_df 是已經存在的資料框架
OIR_columns = [
    'Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng', 'End_Lat',
    'End_Lng', 'Distance(mi)', 'Weather_Timestamp', 'Temperature(F)',
    'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
    'Wind_Speed(mph)', 'Precipitation(in)'
]

# 使用 describe() 方法生成統計描述
statistics_df = data_df[OIR_columns].describe()
print(statistics_df)

# 選擇不在 OIR_columns 中的所有欄位，並將它們轉換為 object 類型
nominal_columns = data_df.loc[:, ~data_df.columns.isin(OIR_columns)].astype("object")

# 使用 describe() 方法生成統計描述
nominal_statistics_df = nominal_columns.describe()
print(nominal_statistics_df)


def showCategoryFig():
    # https://plotly.com/python/sunburst-charts/
    data_category = dict(
        character=["Basic", "Location", "Environment", "Infrastructure", 'ID', 'Severity', 'Start_Time', 'End_Time',
                   'Distance(mi)', 'Description', 'Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng', 'Number', 'Street',
                   'Side', 'City', 'County', 'State', 'Zipcode', 'Country', 'Timezone', 'Airport_Code',
                   'Weather_Timestamp', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
                   'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition',
                   'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight', 'Amenity', 'Bump',
                   'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
                   'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop'],
        parent=["", "", "", "", "Basic", "Basic", "Basic", "Basic", "Basic", "Basic", "Location", "Location",
                "Location", "Location", "Location", "Location", "Location", "Location", "Location", "Location",
                "Location", "Location", "Location", "Location", "Environment", "Environment", "Environment",
                "Environment", "Environment", "Environment", "Environment", "Environment", "Environment", "Environment",
                "Environment", "Environment", "Environment", "Environment", "Infrastructure", "Infrastructure",
                "Infrastructure", "Infrastructure", "Infrastructure", "Infrastructure", "Infrastructure",
                "Infrastructure", "Infrastructure", "Infrastructure", "Infrastructure", "Infrastructure",
                "Infrastructure"],
    )
    category_fig = px.sunburst(
        data_category,
        names='character',
        parents='parent'
    )
    category_fig.update_layout(
        autosize=False,
        width=600,
        height=600,
        title={
            'text': "Data Categorizing",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    category_fig.show()


# Category Visualization
showCategoryFig()

# Irrelevant columns
'''
ID：ID 對於資料集來說是唯一的且無意義的。
Description：我不做文字挖掘，因此它沒用。
Country：所有數據均來自美國，因此所有數據都是相同的。
Weather_Timestamp：天氣觀測記錄的時間戳記。這裡沒啥用。
'''
irrelavant_columns = ['ï»¿ID','Description','Country','Weather_Timestamp']
data_preprocessed_df = data_df.drop(irrelavant_columns, axis=1)

# Replace the empty data with NaN
data_preprocessed_df.replace("", float("NaN"), inplace=True)
data_preprocessed_df.replace(" ", float("NaN"), inplace=True)

# Count missing value(NaN, na, null, None) of each columns, Then transform the result to a pandas dataframe.
count_missing_value = data_preprocessed_df.isna().sum() / data_preprocessed_df.shape[0] * 100
count_missing_value_df = pd.DataFrame(count_missing_value.sort_values(ascending=False), columns=['Missing%'])

# Visualize the percentage(>0) of Missing value in each column.
missing_value_df = count_missing_value_df[count_missing_value_df['Missing%'] > 0]


def visualize_missing_values():
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(missing_value_df)))
    plt.figure(figsize=(15, 10))  # Set the figure size
    missing_value_graph = sns.barplot(y=missing_value_df.index, x="Missing%", data=missing_value_df, palette=colors,
                                      orient="h")
    missing_value_graph.set_title("Percentage Missing value of each feature", fontsize=20)
    missing_value_graph.set_ylabel("Features")
    plt.show()


visualize_missing_values()

## Drop the column with Missing value(>40%)
missing_value_40_df = count_missing_value_df[count_missing_value_df['Missing%'] > 40]
data_preprocessed_df.drop(missing_value_40_df.index, axis=1, inplace=True)
print("Drop the column with Missing value(>40%)")
print(missing_value_40_df)

# Convert Time to datetime64[ns]
data_preprocessed_df['Start_Time'] = pd.to_datetime(data_preprocessed_df['Start_Time'])
data_preprocessed_df['End_Time'] = pd.to_datetime(data_preprocessed_df['End_Time'])

# Display all the missing value
missing_value_df

# Categorize the missing value to numerical and categorical for imputation purpose
numerical_missing = ['Wind_Speed(mph)', 'End_Lng', 'End_Lat', 'Visibility(mi)','Humidity(%)', 'Temperature(F)', 'Pressure(in)']
categorical_missing = ['Weather_Condition','Wind_Direction', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight', 'Side']

# 2.4.1. Drop all NaN/NA/null
# Drop all the instance with NaN/NA/null
data_preprocessed_dropNaN_df = data_preprocessed_df.dropna()
data_preprocessed_dropNaN_df.reset_index(drop=True, inplace=True)

# 2.4.2. Median imputation
# Imputation by corresponding class Median value
data_preprocessed_median_df = data_preprocessed_df.copy()

# For numerical columns
for column_name in numerical_missing:
    data_preprocessed_median_df[column_name] = data_preprocessed_median_df.groupby('Severity')[column_name].transform(lambda x:x.fillna(x.median()))

# # For categorical columns(Majority value imputation)
# https://medium.com/analytics-vidhya/best-way-to-impute-categorical-data-using-groupby-mean-mode-2dc5f5d4e12d
for column_name in categorical_missing:
    data_preprocessed_median_df[column_name] = data_preprocessed_median_df.groupby('Severity')[column_name].transform(lambda x:x.fillna(x.fillna(x.mode().iloc[0])))

# Drop NaN and reset index
data_preprocessed_median_df.dropna(inplace=True)

# 2.4.3. Mean imputation
# Imputation by corresponding class Mean value
data_preprocessed_mean_df = data_preprocessed_df.copy()

# For numerical columns
for column_name in numerical_missing:
    data_preprocessed_mean_df[column_name] = data_preprocessed_mean_df.groupby('Severity')[column_name].transform(
        lambda x: x.fillna(x.mean()))

# For categorical columns(Majority value imputation)
for column_name in categorical_missing:
    data_preprocessed_mean_df[column_name] = data_preprocessed_mean_df.groupby('Severity')[column_name].transform(
        lambda x: x.fillna(x.fillna(x.mode().iloc[0])))

# Drop NaN
data_preprocessed_mean_df.dropna(inplace=True)

# Save these datasets to local
#data_preprocessed_dropNaN_df.to_csv('preprocessed_dropNaN.csv', index=False)
#data_preprocessed_median_df.to_csv('preprocessed_median.csv', index=False)
#data_preprocessed_mean_df.to_csv('preprocessed_mean.csv', index=False)

# Choose the best dataset base on the performance of modeling
data_best_df = data_preprocessed_dropNaN_df.copy()
#data_best_df = data_preprocessed_dropNaN_df[data_preprocessed_dropNaN_df['City'] == 'Orlando'].copy()
#data_best_df = data_preprocessed_median_df[data_preprocessed_dropNaN_df['City'] == 'Orlando'].copy()
#data_best_df = data_preprocessed_mean_df[data_preprocessed_dropNaN_df['City'] == 'Orlando'].copy()

# Reset index
data_best_df.reset_index(inplace=True)