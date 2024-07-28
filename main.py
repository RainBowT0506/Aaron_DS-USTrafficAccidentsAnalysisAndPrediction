# Import packages
import eli5
## Basic data processing
import numpy as np
import pandas as pd

## Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

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
numerical_missing = ['Wind_Speed(mph)', 'Visibility(mi)','Humidity(%)', 'Temperature(F)', 'Pressure(in)']
# numerical_missing = ['Wind_Speed(mph)', 'End_Lng', 'End_Lat', 'Visibility(mi)','Humidity(%)', 'Temperature(F)', 'Pressure(in)']
categorical_missing = ['Weather_Condition','Wind_Direction', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']
# categorical_missing = ['Weather_Condition','Wind_Direction', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight', 'Side']

# 2.4.1. Drop all NaN/NA/null
# Drop all the instance with NaN/NA/null
data_preprocessed_dropNaN_df = data_preprocessed_df.dropna()
data_preprocessed_dropNaN_df.reset_index(drop=True, inplace=True)

# 2.4.2. Median imputation
# Imputation by corresponding class Median value
data_preprocessed_median_df = data_preprocessed_df.copy()

print(data_preprocessed_df.columns)
missing_columns = [col for col in numerical_missing if col not in data_preprocessed_df.columns]
if missing_columns:
    print(f"Missing columns: {missing_columns}")


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

week_accidents_count = data_best_df["Start_Time"].dt.day_name().value_counts()
week_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

year_accidents_count = data_best_df["Start_Time"].dt.year.value_counts()
year_accidents_count_df = pd.DataFrame(year_accidents_count)

# https://hedgescompany.com/automotive-market-research-statistics/auto-mailing-lists-and-marketing/
registered_vehicles_df = pd.DataFrame(data=[264.0, 270.4, 279.1, 284.5, 286.9], columns=['Numbers(million)'],
                                      index=[2016, 2017, 2018, 2019, 2020])

# 3. Data Analysis
# 3.1. Basic Analysis

def visualize_accident_severity_distribution():
    # Count the number of each severity, transform the result to pandas dataframe
    severity_counts = data_best_df["Severity"].value_counts()
    severity_counts_df = pd.DataFrame(severity_counts).reset_index()
    severity_counts_df.columns = ["Severity", "Counts"]

    # Calculate the proportion of each Severity
    severity_counts_df["Percentage"] = severity_counts_df["Counts"] / severity_counts_df["Counts"].sum() * 100

    # Visualize the distribution of accidents severity
    severity_fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "domain"}]])

    severity_fig.add_trace(go.Bar(x=severity_counts_df["Severity"],
                                  y=severity_counts_df["Counts"],
                                  text=severity_counts_df["Counts"],
                                  textposition='outside',
                                  showlegend=False),
                           1, 1)

    severity_fig.add_trace(go.Pie(labels=severity_counts_df["Severity"],
                                  values=severity_counts_df["Percentage"],
                                  showlegend=True),
                           1, 2)

    severity_fig.update_layout(
        height=600,
        width=1500,
        title={
            'text': "The distribution of accidents severity",
            'font': {'size': 24},
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis1_title='Severity',
        yaxis1_title='Counts',
        legend_title_text="Severity"
    )

    severity_fig.update_xaxes(type='category')
    severity_fig.show()


# 3.1.1. What is the distribution of accidents severity?
visualize_accident_severity_distribution()


def visualize_distance_severity_correlation():
    # Calculate the mean distance of each Severity
    mean_distance = data_best_df.groupby('Severity')["Distance(mi)"].mean().round(2)
    mean_distance_df = pd.DataFrame(mean_distance)
    mean_distance_fig = px.bar(mean_distance_df,
                               x=mean_distance_df.index,
                               y="Distance(mi)",
                               labels={"index": "Severity"},
                               text="Distance(mi)")
    mean_distance_fig.update_layout(
        autosize=False,
        width=600,
        height=500,
        title={
            'text': "Mean Distance(mi) of each Severity",
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 24},
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis={
            'type': 'category'
        })
    mean_distance_fig.show()


# 3.1.2. What is the relationship between distance and severity?
visualize_distance_severity_correlation()


# Overview of the US traffic accidents
# https://plotly.com/python/choropleth-maps/
state_accidents_count = data_best_df["State"].value_counts()

# 3.2. Location Analysis
# 3.2.1. What are the top 10 states with the most accidents?
def visualize_top_states_with_most_accidents():
    fig = go.Figure(data=go.Choropleth(locations=state_accidents_count.index,
                                       z=state_accidents_count.values.astype(float),
                                       locationmode="USA-states",
                                       colorscale="Reds",
                                       colorbar_title="Frequency"
                                       ))
    fig.update_layout(
        height=600,
        width=1500,
        title={
            'text': "Frequency distribution of US Accidents",
            'y': 0.9,
            'x': 0.5,
            'font': {'size': 24},
            'xanchor': 'center',
            'yanchor': 'top'},
        geo_scope="usa")
    fig.show()

visualize_top_states_with_most_accidents()

def visualize_top_10_states_with_most_accidents():
    state_accidents_count_top10 = state_accidents_count[:10]
    print("visualize_top_10_states_with_most_accidents")
    print(state_accidents_count_top10)
    state_accidents_count_top10_df = pd.DataFrame(state_accidents_count_top10, columns=['count'])
    state_accidents_count_top10_df.index.name = 'State'
    state_accidents_count_top10_fig = px.bar(state_accidents_count_top10_df,
                                             x=state_accidents_count_top10_df.index,
                                             y="count",
                                             labels={"index": "County", "count": "Counts"},
                                             text="count")
    state_accidents_count_top10_fig.update_layout(
        autosize=False,
        width=1000,
        height=600,
        title={
            'text': "Top 10 States with the Most Accidents",
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 24},
            'xanchor': 'center',
            'yanchor': 'top'})
    state_accidents_count_top10_fig.update_yaxes(categoryorder="total ascending")
    state_accidents_count_top10_fig.show()



# Top 10 States with the Most Accidents
visualize_top_10_states_with_most_accidents()


def visualize_top_10_states_accidents_by_severity():
    # Top 10 States with the Most Accidents in a view of severity
    plt.figure(figsize=(20, 8))
    ax = sns.countplot(x="State",
                       data=data_best_df,
                       order=data_best_df['State'].value_counts()[:10].index,
                       hue='Severity',
                       palette='tab10')
    plt.title("Top 10 States with the Most Accidents", fontsize=22)
    # for p in ax.patches:
    #        ax.annotate(p.get_height(), (p.get_x(), p.get_height()+1000))
    plt.show()


visualize_top_10_states_accidents_by_severity()


def visualize_top_10_counties_with_most_accidents():
    # Top 10 Counties with the Most Accidents
    county_accidents_count = data_best_df["County"].value_counts()
    county_accidents_count_top10 = county_accidents_count[:10]
    print("visualize_top_10_counties_with_most_accidents")
    print(county_accidents_count_top10)
    county_accidents_count_top10_df = pd.DataFrame(county_accidents_count_top10, columns=['count'])
    county_accidents_count_top10_df.index.name = 'County'
    county_accidents_count_top10_fig = px.bar(county_accidents_count_top10_df,
                                              x=county_accidents_count_top10_df.index,
                                              y="count",
                                              labels={"index": "County", "count": "Counts"},
                                              text="count")
    county_accidents_count_top10_fig.update_layout(
        autosize=False,
        width=1000,
        height=600,
        title={
            'text': "Top 10 Counties with the Most Accidents",
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 24},
            'xanchor': 'center',
            'yanchor': 'top'})
    county_accidents_count_top10_fig.update_yaxes(categoryorder="total ascending")
    county_accidents_count_top10_fig.show()

# 3.2.2. What are the top 10 counties with the most accidents?
visualize_top_10_counties_with_most_accidents()


def visualize_top_10_counties_accidents_by_severity():
    plt.figure(figsize=(20, 8))
    ax = sns.countplot(x="County",
                       data=data_best_df,
                       order=data_best_df['County'].value_counts()[:10].index,
                       hue='Severity')
    plt.title("Top 10 Counties with the Most Accidents", fontsize=22)
    for p in ax.patches:
        ax.annotate(p.get_height(), (p.get_x(), p.get_height() + 400))
    plt.show()


# Top 10 Counties with the Most Accidents in a view of severity
visualize_top_10_counties_accidents_by_severity()


def visualize_top_10_cities_with_most_accidents():
    city_accidents_count = data_best_df["City"].value_counts()
    city_accidents_count_top10 = city_accidents_count[:10]
    city_accidents_count_top10_df = pd.DataFrame(city_accidents_count_top10)
    city_accidents_count_top10_fig = px.bar(city_accidents_count_top10_df,
                                            x=city_accidents_count_top10_df.index,
                                            y="count",
                                            labels={"index": "City", "count": "Counts"},
                                            text="count")
    city_accidents_count_top10_fig.update_layout(
        autosize=False,
        width=1000,
        height=600,
        title={
            'text': "Top 10 Cities with the Most Accidents",
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 24},
            'xanchor': 'center',
            'yanchor': 'top'})
    city_accidents_count_top10_fig.update_yaxes(categoryorder="total ascending")
    city_accidents_count_top10_fig.show()


# 3.2.3. What are the top 10 cities with the most accidents?
# Top 10 Cities with the Most Accidents
visualize_top_10_cities_with_most_accidents()


def visualize_top_10_cities_accidents_by_severity():
    plt.figure(figsize=(20, 8))
    ax = sns.countplot(x="City",
                       data=data_best_df,
                       order=data_best_df['City'].value_counts()[:10].index,
                       hue='Severity')
    plt.title("Top 10 Cities with the Most Accidents", fontsize=22)
    for p in ax.patches:
        ax.annotate(p.get_height(), (p.get_x(), p.get_height() + 200))
    plt.show()


# Top 10 Cities with the Most Accidents in a view of severity
visualize_top_10_cities_accidents_by_severity()


def visualize_top_10_streets_with_most_accidents():
    street_accidents_count = data_best_df["Street"].value_counts()
    street_accidents_count_top10 = street_accidents_count[:10]
    street_accidents_count_top10_df = pd.DataFrame(street_accidents_count_top10)
    street_accidents_count_top10_fig = px.bar(street_accidents_count_top10_df,
                                              x=street_accidents_count_top10_df.index,
                                              y="count",
                                              labels={"index": "Street", "count": "Counts"},
                                              text="count")
    street_accidents_count_top10_fig.update_layout(
        autosize=False,
        width=1000,
        height=600,
        title={
            'text': "Top 10 Streets with the Most Accidents",
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 24},
            'xanchor': 'center',
            'yanchor': 'top'})
    street_accidents_count_top10_fig.update_yaxes(categoryorder="total ascending")
    street_accidents_count_top10_fig.show()


# 3.2.4. What is the top 10 Streets with the most accidents?
# Top 10 Streets with the Most Accidents
visualize_top_10_streets_with_most_accidents()


def visualize_top_10_streets_accidents_by_severity():
    plt.figure(figsize=(20, 8))
    ax = sns.countplot(x="Street",
                       data=data_best_df,
                       order=data_best_df['Street'].value_counts()[:10].index,
                       hue='Severity')
    plt.title("Top 10 Streets with the Most Accidents", fontsize=22)
    for p in ax.patches:
        ax.annotate(p.get_height(), (p.get_x(), p.get_height() + 100))
    plt.show()


# Top 10 Streets with the Most Accidents in a view of severity
visualize_top_10_streets_accidents_by_severity()


def visualize_top_10_zipcodes_with_most_accidents():
    zipcode_accidents_count = data_best_df["Zipcode"].value_counts()
    zipcode_accidents_count_top10 = zipcode_accidents_count[:10]
    zipcode_accidents_count_top10_df = pd.DataFrame(zipcode_accidents_count_top10)
    zipcode_accidents_count_top10_fig = px.bar(zipcode_accidents_count_top10_df,
                                               x=zipcode_accidents_count_top10_df.index,
                                               y="count",
                                               labels={"index": "Zipcode", "count": "Counts"},
                                               text="count")
    zipcode_accidents_count_top10_fig.update_layout(
        autosize=False,
        width=1000,
        height=600,
        title={
            'text': "Top 10 Zipcode with the Most Accidents",
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 24},
            'xanchor': 'center',
            'yanchor': 'top'})
    zipcode_accidents_count_top10_fig.update_yaxes(categoryorder="total ascending")
    zipcode_accidents_count_top10_fig.show()


# 3.2.5. What is the top 10 Zipcode with the most accidents?
# Top 10 Zipcode with the Most Accidents
visualize_top_10_zipcodes_with_most_accidents()


def visualize_top_10_zipcodes_accidents_by_severity():
    plt.figure(figsize=(20, 8))
    ax = sns.countplot(x="Zipcode",
                       data=data_best_df,
                       order=data_best_df['Zipcode'].value_counts()[:10].index,
                       hue='Severity')
    plt.title("Top 10 Zipcode with the Most Accidents", fontsize=22)
    for p in ax.patches:
        ax.annotate(p.get_height(), (p.get_x(), p.get_height() + 10))
    plt.show()


# Top 10 Zipcode with the Most Accidents in a view of severity
visualize_top_10_zipcodes_accidents_by_severity()

# 3.2.6. What are the accidents distribution by street Side?
# Accidents distribution by street Side
# Set up the matplotlib figure
def visualize_accident_distribution_by_street_side():
    f, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    # Pie chart
    print("visualize_accident_distribution_by_street_side")
    print(data_best_df)
    data_best_df["Side"].value_counts().plot.pie(autopct="%.1f%%", ylabel='', ax=axes[0])
    sns.countplot(x="Side",
                  data=data_best_df,
                  order=data_best_df['Side'].value_counts().index,
                  hue='Severity',
                  ax=axes[1])
    for p in axes[1].patches:
        axes[1].annotate(p.get_height(), (p.get_x() + 0.05, p.get_height() + 100))
    # Common title
    plt.suptitle("Accidents distribution by street Side", y=0.95, fontsize=20)
    plt.show()

visualize_accident_distribution_by_street_side()

def visualize_yearly_accident_change():
    year_accidents_count_fig = px.bar(year_accidents_count,
                                      x=year_accidents_count.index,
                                      y="count",
                                      labels={"index": "Year", "count": "Counts"},
                                      text="count")
    year_accidents_count_fig.update_layout(
        autosize=False,
        width=800,
        height=600,
        title={
            'text': "Accidents yearly change",
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 24},
            'xanchor': 'center',
            'yanchor': 'top'})
    year_accidents_count_fig.show()


# 3.3. Time Analysis
# 3.3.1. Accidents yearly change

# Accidents yearly change
visualize_yearly_accident_change()

def visualize_yearly_change_in_registered_vehicles():
    registered_vehicles_fig = px.bar(registered_vehicles_df,
                                     x=registered_vehicles_df.index,
                                     y="Numbers(million)",
                                     labels={"index": "Year"},
                                     text="Numbers(million)")
    registered_vehicles_fig.update_layout(
        autosize=False,
        width=800,
        height=600,
        title={
            'text': "Registered motor vehicles yearly change",
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 24},
            'xanchor': 'center',
            'yanchor': 'top'})
    registered_vehicles_fig.show()


# The number of registered motor vehicles
visualize_yearly_change_in_registered_vehicles()


def visualize_accident_vehicle_change_comparison():
    print("visualize_accident_vehicle_change_comparison")
    # 確認 year_accidents_count_df 的結構和列名
    year_accidents_pct_df = year_accidents_count_df.sort_index().pct_change().rename(
        columns={'count': 'Accidents'})  # 確認並重命名列名為 'Accidents'
    print("year_accidents_pct_df:")
    print(year_accidents_pct_df)

    # 確認 registered_vehicles_df 的結構和列名
    registered_vehicles_pct_df = registered_vehicles_df.pct_change().rename(columns={'Numbers(million)': 'Vehicles'})
    print("registered_vehicles_pct_df:")
    print(registered_vehicles_pct_df)

    # 合併數據框
    comparison_pct_df = pd.concat([year_accidents_pct_df, registered_vehicles_pct_df], axis=1).dropna()
    print("comparison_pct_df:")
    print(comparison_pct_df)

    # 繪製圖表
    comparison_pct_fig = go.Figure()
    comparison_pct_fig.add_trace(go.Scatter(x=comparison_pct_df.index, y=comparison_pct_df['Accidents'],
                                            mode='lines+markers',
                                            name='Accidents'))
    comparison_pct_fig.add_trace(go.Scatter(x=comparison_pct_df.index, y=comparison_pct_df['Vehicles'],
                                            mode='lines+markers',
                                            name='Vehicles'))
    comparison_pct_fig.update_layout(
        autosize=False,
        width=800,
        height=600,
        title={
            'text': "Accidents Vs Vehicles (Number)",
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 24},
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title='Year',
        yaxis_title='Percentage Rate(%)')
    comparison_pct_fig.update_xaxes(type='category')
    comparison_pct_fig.show()


# Calcute and Concat yearly accidents change rate and yearly registered vehicles change rate, then compare them in one graph.
visualize_accident_vehicle_change_comparison()


def visualize_monthly_accident_change():
    # https://plotly.com/python/categorical-axes/
    month_accidents_count = data_best_df["Start_Time"].dt.month.value_counts()
    print("visualize_monthly_accident_change")
    print(month_accidents_count)
    month_accidents_count_df = month_accidents_count.reset_index()
    month_accidents_count_df.columns = ['Month', 'Count']
    month_accidents_count_fig = px.bar(month_accidents_count_df,
                                       x='Month',
                                       y='Count',
                                       labels={"Month": "Month", "Count": "Counts"},
                                       text='Count')
    month_accidents_count_fig.update_layout(
        autosize=False,
        width=1000,
        height=500,
        title={
            'text': "Accidents Monthly Change",
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 24},
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis={
            'type': 'category'
        })
    month_accidents_count_fig.show()


# 3.3.2. Month
# Extract Month
visualize_monthly_accident_change()

def visualize_weekly_accident_change():
    # 將 week_accidents_count 轉換為數據框並重置索引
    week_accidents_count_df = pd.DataFrame(week_accidents_count).reset_index()
    week_accidents_count_df.columns = ['Week', 'Count']  # 重命名列

    print("visualize_weekly_accident_change")
    print(week_accidents_count_df)

    # 使用 Plotly 繪製條形圖
    week_accidents_count_fig = px.bar(week_accidents_count_df,
                                      x='Week',
                                      y='Count',
                                      labels={"Week": "Week", "Count": "Counts"},
                                      text='Count')
    week_accidents_count_fig.update_layout(
        autosize=False,
        width=800,
        height=500,
        title={
            'text': "Accidents Weekly Change",
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 24},
            'xanchor': 'center',
            'yanchor': 'top'})
    # week_accidents_count_fig.update_xaxes(categoryorder = "total ascending")
    week_accidents_count_fig.show()


# 3.3.3. Week
# Extract Week
visualize_weekly_accident_change()


def visualize_monthly_weekly_accident_change():
    data_best_df["Month"] = data_best_df["Start_Time"].dt.month
    data_best_df["Week"] = data_best_df["Start_Time"].dt.day_name()
    data_best_df["Hour"] = data_best_df["Start_Time"].dt.hour
    # Monthly view with weeks
    data_best_df.groupby('Month')['Week'].value_counts().unstack()[week_order].plot.bar(
        figsize=(20, 8),
        ylabel='Counts',
        width=.8
    )
    plt.title("Accidents Monthly change in a view of week", fontsize=22)
    plt.show()


# Transform Month/week/Hour to different features
visualize_monthly_weekly_accident_change()


# 3.3.4. Hour
# Extract Hour (Weekday)
def visualize_hourly_weekday_accident_change():
    weekdays_lst = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    hour_accidents_weekdays_count = data_best_df[data_best_df['Week'].isin(weekdays_lst)]["Start_Time"].dt.hour
    hour_accidents_weekdays_count_df = pd.DataFrame(hour_accidents_weekdays_count.value_counts()).reset_index()
    hour_accidents_weekdays_count_df.columns = ['Hour', 'Count']  # 重命名列
    hour_accidents_weekdays_count_df.sort_values(by='Hour', inplace=True)

    print("visualize_hourly_weekday_accident_change")
    print(hour_accidents_weekdays_count_df)

    # 使用 Plotly 繪製條形圖
    hour_accidents_weekdays_count_fig = px.bar(hour_accidents_weekdays_count_df,
                                               x='Hour',
                                               y='Count',
                                               labels={"Hour": "Hour", "Count": "Counts"},
                                               text='Count')
    hour_accidents_weekdays_count_fig.update_layout(
        autosize=False,
        width=1000,
        height=500,
        title={
            'text': "Accidents Hourly Change (Weekdays)",
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 24},
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis={
            'type': 'category'
        })
    hour_accidents_weekdays_count_fig.show()

visualize_hourly_weekday_accident_change()


def visualize_hourly_weekend_accident_change():
    weekend_lst = ['Saturday', 'Sunday']
    hour_accidents_weekend_count = data_best_df[data_best_df['Week'].isin(weekend_lst)]["Start_Time"].dt.hour
    hour_accidents_weekend_count_df = pd.DataFrame(hour_accidents_weekend_count.value_counts())
    hour_accidents_weekend_count_df.sort_index(inplace=True)
    print("visualize_hourly_weekend_accident_change")
    print(hour_accidents_weekend_count_df)
    hour_accidents_weekend_count_fig = px.bar(hour_accidents_weekend_count_df,
                                              x=hour_accidents_weekend_count_df.index,
                                              y="count",
                                              labels={"index": "Hour", "count": "Counts"},
                                              text="count")
    hour_accidents_weekend_count_fig.update_layout(
        autosize=False,
        width=1000,
        height=500,
        title={
            'text': "Accidents hourly change(Weekend)",
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 24},
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis={
            'type': 'category'
        })
    hour_accidents_weekend_count_fig.show()


# Extract Hour (Weekend)
visualize_hourly_weekend_accident_change()


def visualize_weekly_hourly_accident_change():
    data_best_df.groupby('Week')['Hour'].value_counts().unstack().reindex(week_order).plot.bar(
        figsize=(22, 8),
        ylabel='Counts',
        width=.9
    )
    plt.title("Accidents Weekly change in a view of hour", fontsize=22)
    plt.show()


# Weekly view with hours
visualize_weekly_hourly_accident_change()


def visualize_mean_duration_per_severity():
    data_best_df["Duration"] = (data_best_df['End_Time'] - data_best_df['Start_Time']).dt.total_seconds() / 3600
    mean_duration = data_best_df.groupby('Severity')["Duration"].mean().round(2)
    mean_duration_df = pd.DataFrame(mean_duration)
    mean_duration_fig = px.bar(mean_duration_df,
                               x=mean_duration_df.index,
                               y="Duration",
                               labels={"index": "Severity"},
                               text="Duration")
    mean_duration_fig.update_layout(
        autosize=False,
        width=600,
        height=500,
        title={
            'text': "Mean Duration of each Severity",
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 24},
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis={
            'type': 'category'
        })
    mean_duration_fig.show()


# 3.3.5. Duration and Severity
# Calculate the mean Duration of each Severity
visualize_mean_duration_per_severity()


# 3.4. Environment Analysis
# 3.4.1. Top 15 Weather_Condition
def visualize_weather_conditions():
    fig = plt.figure(figsize=(15, 10))
    sns.countplot(y='Weather_Condition',
                  data=data_best_df,
                  order=data_best_df['Weather_Condition'].value_counts()[:15].index) \
        .set_title("Top 15 Weather_Condition", fontsize=22)
    plt.show()


# Weather condition
visualize_weather_conditions()


def visualize_weather_severity_means():
    weather_mean_severity = data_best_df.groupby('Weather_Condition')['Severity'].mean().sort_values(ascending=False)
    weather_mean_severity_df = pd.DataFrame(weather_mean_severity[:25])
    plt.figure(figsize=(15, 10))  # Set the figure size
    weather_mean_severity_graph = sns.barplot(y=weather_mean_severity_df.index, x="Severity",
                                              data=weather_mean_severity_df, orient="h")
    weather_mean_severity_graph.set_title("Weather Condition with mean of the severity", fontsize=20)
    weather_mean_severity_graph.set_ylabel("Weather_Condition")


# Weather condition by mean of the Severity
visualize_weather_severity_means()

# 3.4.2. Top 15 Wind_Direction
def visualize_wind_directions():
    fig = plt.figure(figsize=(15, 10))
    sns.countplot(y='Wind_Direction',
                  data=data_best_df,
                  order=data_best_df['Wind_Direction'].value_counts()[:15].index) \
        .set_title("Top 15 Wind_Direction", fontsize=22)
    plt.show()


visualize_wind_directions()


def visualize_wind_severity_means():
    wind_mean_severity = data_best_df.groupby('Wind_Direction')['Severity'].mean().sort_values(ascending=False)
    wind_mean_severity_df = pd.DataFrame(wind_mean_severity)
    plt.figure(figsize=(15, 10))  # Set the figure size
    wind_mean_severity_graph = sns.barplot(y=wind_mean_severity_df.index, x="Severity", data=wind_mean_severity_df,
                                           orient="h")
    wind_mean_severity_graph.set_title("Wind Direction with Mean Severity", fontsize=20)
    wind_mean_severity_graph.set_xlabel("Mean Severity")
    wind_mean_severity_graph.set_ylabel("Wind Direction")
    plt.show()


# Wind direction by mean of the Severity
visualize_wind_severity_means()

# 3.4.3. Environment Attribute(numerical) distribution and the relationship with Severity
def visualize_weather_factors_distributions():
    f, axes = plt.subplots(5, 2, figsize=(20, 30))
    sns.distplot(data_best_df['Temperature(F)'], ax=axes[0, 0]).set_title('Temperature(F) Distribution')
    data_best_df["Severity"].groupby(pd.cut(data_best_df['Temperature(F)'], 10)).mean().plot(ylabel='Mean Severity',
                                                                                             title='Mean Severity of Temperature(F)',
                                                                                             ax=axes[0, 1])
    sns.distplot(data_best_df['Humidity(%)'], ax=axes[1, 0]).set_title('Humidity(%) Distribution')
    data_best_df["Severity"].groupby(pd.cut(data_best_df['Humidity(%)'], 10)).mean().plot(ylabel='Mean Severity',
                                                                                          title='Mean Severity of Humidity(%)',
                                                                                          ax=axes[1, 1])
    sns.distplot(data_best_df['Pressure(in)'], ax=axes[2, 0]).set_title('Pressure(in) Distribution')
    data_best_df["Severity"].groupby(pd.cut(data_best_df['Pressure(in)'], 10)).mean().plot(ylabel='Mean Severity',
                                                                                           title='Mean Severity of Pressure(in)',
                                                                                           ax=axes[2, 1])
    sns.distplot(data_best_df['Visibility(mi)'], ax=axes[3, 0]).set_title('Visibility(mi) Distribution')
    data_best_df["Severity"].groupby(pd.cut(data_best_df['Visibility(mi)'], 10)).mean().plot(ylabel='Mean Severity',
                                                                                             title='Mean Severity of Visibility(mi)',
                                                                                             ax=axes[3, 1])
    sns.distplot(data_best_df['Wind_Speed(mph)'], ax=axes[4, 0]).set_title('Wind_Speed(mph) Distribution')
    data_best_df["Severity"].groupby(pd.cut(data_best_df['Wind_Speed(mph)'], 10)).mean().plot(ylabel='Mean Severity',
                                                                                              title='Mean Severity of Wind_Speed(mph)',
                                                                                              ax=axes[4, 1])
    plt.suptitle("Temperature, Humidity, Pressure, Visibility and Wind Speed - Distribution && Mean Severity", y=0.95,
                 fontsize=20)
    plt.plot()


# Set up the matplotlib figure
visualize_weather_factors_distributions()


def visualize_accidents_sunrise_sunset():
    # Accidents distribution by Sunrise && Sunset
    # Set up the matplotlib figure
    f, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    # Pie chart
    data_best_df["Sunrise_Sunset"].value_counts().plot.pie(autopct="%.1f%%", ylabel='', ax=axes[0])
    sns.countplot(x="Sunrise_Sunset",
                  data=data_best_df,
                  order=data_best_df['Sunrise_Sunset'].value_counts().index,
                  hue='Severity',
                  ax=axes[1])
    for p in axes[1].patches:
        axes[1].annotate(p.get_height(), (p.get_x() + 0.025, p.get_height() + 100))
    # Common title
    plt.suptitle("Accidents distribution by Sunrise && Sunset", y=0.95, fontsize=20)
    plt.show()


# 3.4.4. Accidents distribution by Sunrise && Sunset
visualize_accidents_sunrise_sunset()



# 3.5. Infrastructure Analysis
def visualize_infrastructure_feature_distribution():
    f, axes = plt.subplots(13, 2, figsize=(20, 80))
    infrastructure_features = ['Traffic_Signal', 'Crossing', 'Station', 'Amenity', 'Bump', 'Give_Way', 'Junction',
                               'No_Exit', 'Railway', 'Roundabout', 'Stop', 'Traffic_Calming', 'Turning_Loop']
    for infrastructure_feature, number in zip(infrastructure_features, range(0, 13)):
        data_best_df[infrastructure_feature].value_counts().plot.pie(autopct="%.2f%%", ylabel='',
                                                                     ax=axes[number, 0]).set_title(
            f'{infrastructure_feature} Distribution')
        sns.countplot(x=infrastructure_feature,
                      data=data_best_df,
                      order=data_best_df[infrastructure_feature].value_counts().index,
                      hue='Severity',
                      ax=axes[number, 1]).set_title(f'{infrastructure_feature} Analysis')
        # Add number on corresponding bar
        for p in axes[number, 1].patches:
            axes[number, 1].annotate(p.get_height(), (p.get_x() + 0.025, p.get_height() + 100))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle("Severity Distribution by Infrastructure Features", y=0.98, fontsize=20)
    plt.show()

# Show the severity distribution of each category(True/False)
visualize_infrastructure_feature_distribution()


# 4. Feature Engineering

## Only choose a city because of the resources limitation.
data_best_df = data_preprocessed_dropNaN_df[data_preprocessed_dropNaN_df['City'] == 'Orlando'].copy()
#data_best_df = data_preprocessed_median_df[data_preprocessed_dropNaN_df['City'] == 'Orlando'].copy()
#data_best_df = data_preprocessed_mean_df[data_preprocessed_dropNaN_df['City'] == 'Orlando'].copy()

# Reset index
data_best_df.reset_index(inplace=True)

# 4.1. Feature choosing
# Choose relevant features
relevant_features = ['Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng','Side',
       'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
       'Wind_Direction', 'Wind_Speed(mph)', 'Weather_Condition', 'Amenity',
       'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
       'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
       'Turning_Loop', 'Sunrise_Sunset']
data_modelling_df = data_best_df[relevant_features].copy()

# Duration = End_Time - Start_Time; Create a new feature for modeling.
data_modelling_df['Duration'] = (data_modelling_df['End_Time'] - data_modelling_df['Start_Time']).dt.total_seconds() / 3600
data_modelling_df.drop('End_Time', axis=1, inplace=True)

# Transform Month/week/Hour to different features
data_modelling_df["Month"] = data_modelling_df["Start_Time"].dt.month
data_modelling_df["Week"] = data_modelling_df["Start_Time"].dt.dayofweek
data_modelling_df["Hour"] = data_modelling_df["Start_Time"].dt.hour
data_modelling_df.drop("Start_Time", axis=1, inplace=True)

# 4.2. One Hot Encoding
# Select features that are suitable for One Hot Encoding
one_hot_features = ['Wind_Direction', 'Weather_Condition']

# Wind_Direction Categorizing
data_modelling_df.loc[data_modelling_df['Wind_Direction'].str.startswith('C'), 'Wind_Direction'] = 'C' #Calm
data_modelling_df.loc[data_modelling_df['Wind_Direction'].str.startswith('E'), 'Wind_Direction'] = 'E' #East, ESE, ENE
data_modelling_df.loc[data_modelling_df['Wind_Direction'].str.startswith('W'), 'Wind_Direction'] = 'W' #West, WSW, WNW
data_modelling_df.loc[data_modelling_df['Wind_Direction'].str.startswith('S'), 'Wind_Direction'] = 'S' #South, SSW, SSE
data_modelling_df.loc[data_modelling_df['Wind_Direction'].str.startswith('N'), 'Wind_Direction'] = 'N' #North, NNW, NNE
data_modelling_df.loc[data_modelling_df['Wind_Direction'].str.startswith('V'), 'Wind_Direction'] = 'V' #Variable

# Weather_Condition Categorizing
# Fair, Cloudy, Clear, Overcast, Snow, Haze, Rain, Thunderstorm, Windy, Hail, Thunder, Dust, Tornado
data_modelling_df['Weather_Fair'] = np.where(data_modelling_df['Weather_Condition'].str.contains('Fair', case=False, na = False), 1, 0)
data_modelling_df['Weather_Cloudy'] = np.where(data_modelling_df['Weather_Condition'].str.contains('Cloudy', case=False, na = False), 1, 0)
data_modelling_df['Weather_Clear'] = np.where(data_modelling_df['Weather_Condition'].str.contains('Clear', case=False, na = False), 1, 0)
data_modelling_df['Weather_Overcast'] = np.where(data_modelling_df['Weather_Condition'].str.contains('Overcast', case=False, na = False), 1, 0)
data_modelling_df['Weather_Snow'] = np.where(data_modelling_df['Weather_Condition'].str.contains('Snow|Wintry|Sleet', case=False, na = False), 1, 0)
data_modelling_df['Weather_Haze'] = np.where(data_modelling_df['Weather_Condition'].str.contains('Smoke|Fog|Mist|Haze', case=False, na = False), 1, 0)
data_modelling_df['Weather_Rain'] = np.where(data_modelling_df['Weather_Condition'].str.contains('Rain|Drizzle|Showers', case=False, na = False), 1, 0)
data_modelling_df['Weather_Thunderstorm'] = np.where(data_modelling_df['Weather_Condition'].str.contains('Thunderstorms|T-Storm', case=False, na = False), 1, 0)
data_modelling_df['Weather_Windy'] = np.where(data_modelling_df['Weather_Condition'].str.contains('Windy|Squalls', case=False, na = False), 1, 0)
data_modelling_df['Weather_Hail'] = np.where(data_modelling_df['Weather_Condition'].str.contains('Hail|Ice Pellets', case=False, na = False), 1, 0)
data_modelling_df['Weather_Thunder'] = np.where(data_modelling_df['Weather_Condition'].str.contains('Thunder', case=False, na = False), 1, 0)
data_modelling_df['Weather_Dust'] = np.where(data_modelling_df['Weather_Condition'].str.contains('Dust', case=False, na = False), 1, 0)
data_modelling_df['Weather_Tornado'] = np.where(data_modelling_df['Weather_Condition'].str.contains('Tornado', case=False, na = False), 1, 0)

# Transform the one-hot features, then delete them
onehot_df = pd.get_dummies(data_modelling_df['Wind_Direction'], prefix='Wind')
data_modelling_df = pd.concat([data_modelling_df, onehot_df], axis=1)
data_modelling_df.drop(one_hot_features, axis=1, inplace=True)

# 4.3. Label Encoding
# Select features that are suitable for Label Encoding
label_encoding_features = ['Side', 'Amenity','Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway','Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal','Turning_Loop', 'Sunrise_Sunset']

# Label Encoding
for feature in label_encoding_features:
    data_modelling_df[feature] = LabelEncoder().fit_transform(data_modelling_df[feature])

data_modelling_df


# 4.4. Correlation Analysis
# Display the correlation table for continuous features
# https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
def style_corr(v, props=''):
    return props if (v < -0.4 or v > 0.4) and v != 1 else None

continuous_feature = ['Start_Lat', 'Start_Lng', 'Temperature(F)','Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Duration']
data_modelling_df[continuous_feature].corr().style.applymap(style_corr, props='color:red;')


def visualize_feature_correlation():
    plt.figure(figsize=(12, 9))
    sns.heatmap(data_modelling_df[continuous_feature].corr(), cmap="coolwarm", annot=True, fmt='.3f').set_title(
        'Pearson Correlation for continuous features', fontsize=22)
    plt.show()


# Show the heatmap
visualize_feature_correlation()


# Find the data with all the same value
unique_counts = data_modelling_df.drop(continuous_feature, axis=1).astype("object").describe().loc['unique']
feature_all_same = list(unique_counts[unique_counts == 1].index)
data_modelling_df.drop(feature_all_same, axis=1, inplace=True)


# Display the correlation table for categorical features
data_modelling_df.drop(continuous_feature, axis=1).corr(method='spearman').style.applymap(style_corr, props='color:red;')


def visualize_categorical_feature_correlation():
    plt.figure(figsize=(35, 20))
    sns.heatmap(data_modelling_df.drop(continuous_feature, axis=1).corr(method='spearman'), cmap="coolwarm", annot=True,
                fmt='.3f').set_title('Spearman Correlation for categorical features', fontsize=22)
    plt.show()


# Show the heatmap
visualize_categorical_feature_correlation()


# 5. Modelling
## 5.1. Workflow Demonstration
## 5.1.1. End-Time Prediction
# Train/Test Split
X_reg = data_modelling_df.drop(["Severity", "Duration"], axis=1)
Y_reg = data_modelling_df.Duration
x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, Y_reg, test_size = 0.3, random_state=0)
print(f'Train Reg: {x_train_reg.shape} \n Test Reg: {x_test_reg.shape}')
reg_feature_names = x_train_reg.columns.tolist()

# Store the information for Adj R2.
# Adjusted R-squared can be negative when R-squared is close to zero.
# Adjusted R-squared value always be less than or equal to R-squared value.
# http://net-informations.com/ds/psa/adjusted.htm
'''
Description: Calculate Adj R2 for Regression
Args:
    test_data: The test dataset(only feature)
    r2_score: r2 score of the model
Return: adj_r2_score
'''
def adj_r2(test_data, r2_score):
    records_num = test_data.shape[0]
    feature_num = test_data.shape[1]
    adj_r2_score = 1 - ((records_num - 1) / (records_num - feature_num - 1) * (1 - r2_score))
    return adj_r2_score

# 5.1.1.1. Linear Regression
# Linear Regression
lreg = LinearRegression(fit_intercept=False, normalize=True).fit(x_train_reg, y_train_reg)
lreg_predictions = lreg.predict(x_test_reg)
lreg_rmse = mean_squared_error(y_test_reg, lreg_predictions, squared=False)
lreg_r2 = r2_score(y_test_reg, lreg_predictions)
lreg_adj_r2 = adj_r2(x_test_reg, lreg_r2)
print(f'RMSE: {lreg_rmse} \n R2: {lreg_r2} \n Adj_R2: {lreg_adj_r2}')
# Show feature importance as a table
eli5.show_weights(lreg, feature_names = reg_feature_names)


# 5.1.1.2. Support Vector Machine
# SVR
sv_reg = SVR(C=10, gamma=1).fit(x_train_reg, y_train_reg)
sv_reg_predictions = sv_reg.predict(x_test_reg)
sv_reg_rmse = mean_squared_error(y_test_reg, sv_reg_predictions, squared=False)
sv_reg_r2 = r2_score(y_test_reg, sv_reg_predictions)
sv_reg_adj_r2 = adj_r2(x_test_reg, sv_reg_r2)
print(f'RMSE: {sv_reg_rmse} \n R2: {sv_reg_r2} \n Adj_R2: {sv_reg_adj_r2}')
# Show feature importance as a table
#eli5.show_weights(sv_reg, feature_names = reg_feature_names)

# 5.1.1.3. Decision Tree
# Decision Tree
dt_reg = DecisionTreeRegressor(random_state=0).fit(x_train_reg, y_train_reg)
dt_reg_predictions = dt_reg.predict(x_test_reg)
dt_reg_rmse = mean_squared_error(y_test_reg, dt_reg_predictions, squared=False)
dt_reg_r2 = r2_score(y_test_reg, dt_reg_predictions)
dt_reg_adj_r2 = adj_r2(x_test_reg, dt_reg_r2)
print(f'RMSE: {dt_reg_rmse} \n R2: {dt_reg_r2} \n Adj_R2: {dt_reg_adj_r2}')
# Show feature importance as a table
eli5.show_weights(dt_reg, feature_names = reg_feature_names)

# Get the feature importance as a dataframe
def visualize_decision_tree_feature_importance():
    # Get the feature importance as a dataframe
    dt_reg_importances_df = pd.DataFrame(pd.Series(dt_reg.feature_importances_, index=X_reg.columns),
                                         columns=['Importance']).sort_values('Importance', ascending=False)
    # Visualize the feature importance of the trained tree
    plt.figure(figsize=(15, 10))
    missing_value_graph = sns.barplot(y=dt_reg_importances_df.index, x="Importance", data=dt_reg_importances_df,
                                      orient="h")
    missing_value_graph.set_title("Feature importance by Decision Tree Regression", fontsize=20)
    missing_value_graph.set_ylabel("Features")
    plt.show()


visualize_decision_tree_feature_importance()

# 5.1.1.4. Gradient Boosting Tree
# Gradient Boosting Regression
gbt_reg = GradientBoostingRegressor(learning_rate=0.1, max_depth=20, min_impurity_decrease=0.1, min_samples_leaf=10, n_estimators=200, random_state=0).fit(x_train_reg, y_train_reg)
gbt_reg_predictions = gbt_reg.predict(x_test_reg)
gbt_reg_rmse = mean_squared_error(y_test_reg, gbt_reg_predictions, squared=False)
gbt_reg_r2 = r2_score(y_test_reg, gbt_reg_predictions)
gbt_reg_adj_r2 = adj_r2(x_test_reg, gbt_reg_r2)
print(f'RMSE: {gbt_reg_rmse} \n R2: {gbt_reg_r2} \n Adj_R2: {gbt_reg_adj_r2}')
# Show feature importance as a table
eli5.show_weights(gbt_reg, feature_names = reg_feature_names)


def visualize_gbt_feature_importance():
    # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    # Calculate Standard Deviation of each feature for all the trees
    gbt_reg_importances_std = np.std([tree[0].feature_importances_ for tree in gbt_reg.estimators_], axis=0)
    gbt_reg_importances = pd.Series(gbt_reg.feature_importances_, index=X_reg.columns)
    gbt_reg_importances_df = pd.DataFrame(gbt_reg_importances, columns=['Importance'])
    gbt_reg_importances_df['Std'] = gbt_reg_importances_std
    gbt_reg_importances_df.sort_values('Importance', ascending=True, inplace=True)
    fig, ax = plt.subplots(figsize=(15, 10))
    gbt_reg_importances_df['Importance'].plot.barh(xerr=gbt_reg_importances_df['Std'], color='cornflowerblue', ax=ax)
    ax.set_title("Feature importances using MDI of Gradient Boosting Regression", fontsize=22)
    ax.set_xlabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()


visualize_gbt_feature_importance()

# 5.1.1.5. Random Forest
# Random Forest Regression
rf_reg = RandomForestRegressor(random_state=0).fit(x_train_reg, y_train_reg)
rf_reg_predictions = rf_reg.predict(x_test_reg)
rf_reg_rmse = mean_squared_error(y_test_reg, rf_reg_predictions, squared=False)
rf_reg_r2 = r2_score(y_test_reg, rf_reg_predictions)
rf_reg_adj_r2 = adj_r2(x_test_reg, rf_reg_r2)
print(f'RMSE: {rf_reg_rmse} \n R2: {rf_reg_r2} \n Adj_R2: {rf_reg_adj_r2}')
# Show feature importance as a table
eli5.show_weights(rf_reg, feature_names = reg_feature_names)


def visualize_rf_feature_importance():
    rf_reg_importances_std = np.std([tree.feature_importances_ for tree in rf_reg.estimators_], axis=0)
    rf_reg_importances = pd.Series(rf_reg.feature_importances_, index=X_reg.columns)
    rf_reg_importances_df = pd.DataFrame(rf_reg_importances, columns=['Importance'])
    rf_reg_importances_df['Std'] = rf_reg_importances_std
    rf_reg_importances_df.sort_values('Importance', ascending=True, inplace=True)
    fig, ax = plt.subplots(figsize=(15, 10))
    rf_reg_importances_df['Importance'].plot.barh(xerr=rf_reg_importances_df['Std'], color='cornflowerblue', ax=ax)
    ax.set_title("Feature importances using MDI of Random Forest Regression", fontsize=22)
    ax.set_xlabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()


# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
# Calculate Standard Deviation of each feature for all the trees
visualize_rf_feature_importance()


# 5.1.1.6. XGBoost
# XGB Regression
xgb_reg = XGBRegressor(learning_rate=0.1, max_depth=30, n_estimators=50, random_state=0).fit(x_train_reg, y_train_reg)
xgb_reg_predictions = xgb_reg.predict(x_test_reg)
xgb_reg_rmse = mean_squared_error(y_test_reg, xgb_reg_predictions, squared=False)
xgb_reg_r2 = r2_score(y_test_reg, xgb_reg_predictions)
xgb_reg_adj_r2 = adj_r2(x_test_reg, xgb_reg_r2)
print(f'RMSE: {xgb_reg_rmse} \n R2: {xgb_reg_r2} \n Adj_R2: {xgb_reg_adj_r2}')
# Show feature importance as a table
eli5.show_weights(xgb_reg, feature_names = reg_feature_names)


def visualize_xgb_feature_importance():
    xgb_reg_importances_df = pd.DataFrame(pd.Series(xgb_reg.feature_importances_, index=X_reg.columns),
                                          columns=['Importance']).sort_values('Importance', ascending=False)
    # Visualize the feature importance of the trained tree
    plt.figure(figsize=(15, 10))
    missing_value_graph = sns.barplot(y=xgb_reg_importances_df.index, x="Importance", data=xgb_reg_importances_df,
                                      orient="h")
    missing_value_graph.set_title("Feature importance by XGB Regression", fontsize=20)
    missing_value_graph.set_ylabel("Features")
    plt.show()


# Get the feature importance as a dataframe
visualize_xgb_feature_importance()

# 5.1.1.7. Multi-layer Perceptron Regression
# Multi-layer Perceptron regressor.
mlpr_reg = MLPRegressor(learning_rate='invscaling', hidden_layer_sizes=(50, 75, 100), random_state=0).fit(x_train_reg, y_train_reg)
mlpr_reg_predictions = mlpr_reg.predict(x_test_reg)
mlpr_reg_rmse = mean_squared_error(y_test_reg, mlpr_reg_predictions, squared=False)
mlpr_reg_r2 = r2_score(y_test_reg, mlpr_reg_predictions)
mlpr_reg_adj_r2 = adj_r2(x_test_reg, mlpr_reg_r2)
print(f'RMSE: {mlpr_reg_rmse} \n R2: {mlpr_reg_r2} \n Adj_R2: {mlpr_reg_adj_r2}')


# 5.1.1.8. Model Comparison
# Gather all the Regression performance in one table
reg_results = pd.DataFrame([(lreg_rmse, lreg_r2, lreg_adj_r2), (sv_reg_rmse, sv_reg_r2, sv_reg_adj_r2), (dt_reg_rmse, dt_reg_r2, dt_reg_adj_r2), (gbt_reg_rmse, gbt_reg_r2, gbt_reg_adj_r2), (rf_reg_rmse, rf_reg_r2, rf_reg_adj_r2), (xgb_reg_rmse, xgb_reg_r2, xgb_reg_adj_r2), (mlpr_reg_rmse, mlpr_reg_r2, mlpr_reg_adj_r2)],
             columns=['RMSE','R2','Adj_R2'],
             index= ['Linear Regression',
                    'Support Vector Machine',
                    'Decision Tree',
                    'Gradient Boosting Tree',
                    'Random Forest',
                    'XGBoost',
                    'Multi-layer Perceptron'])
reg_results.sort_values(by=['RMSE'])


# 5.1.2. Severity Prediction
# Train/Test Split
X_cla = data_modelling_df.drop("Severity", axis=1)
Y_cla = data_modelling_df.Severity
x_train_cla, x_test_cla, y_train_cla, y_test_cla = train_test_split(X_cla, Y_cla, test_size = 0.3, random_state=0, stratify=Y_cla)
print(f'Train Cla: {x_train_cla.shape} \n Test Cla: {x_test_cla.shape}')
cla_feature_names = x_train_cla.columns.tolist()



# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html#sklearn.metrics.plot_confusion_matrix
# normalize must be one of {'true', 'pred', 'all', None}
'''
Description: Plot the confusion matrix
Args:
    classifier: The classifier
Return: None
'''
def draw_confusion_matrix(classifier):
    fig, ax = plt.subplots(figsize=(12, 6))
    ConfusionMatrixDisplay(classifier, x_test_cla, y_test_cla, cmap=plt.cm.Blues, normalize=None, ax=ax)
    ax.set_title("Confusion Matrix", fontsize = 15)
    plt.show()

# 5.1.2.1. Logistic Regression
# Logistic Regression
logistic_reg = LogisticRegression(C=10, fit_intercept=False, solver='liblinear')
logistic_reg.fit(x_train_cla, y_train_cla)
logistic_reg_predictions = logistic_reg.predict(x_test_cla)
logistic_reg_results = classification_report(y_test_cla, logistic_reg_predictions, zero_division=True, output_dict=True)

# Confusion matrix and Classification report
draw_confusion_matrix(logistic_reg)
print(classification_report(y_test_cla, logistic_reg_predictions, zero_division=True))

# balanced_accuracy
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score
logistic_balanced_accuracy = balanced_accuracy_score(y_test_cla, logistic_reg_predictions)
print(f'balanced_accuracy: {logistic_balanced_accuracy}')

# ROC_AUC score
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
logistic_roc_ovo_macro = roc_auc_score(y_test_cla, logistic_reg.predict_proba(x_test_cla), multi_class='ovo', average='macro') #Insensitive to class imbalance when average == 'macro'
logistic_roc_ovr_weighted = roc_auc_score(y_test_cla, logistic_reg.predict_proba(x_test_cla), multi_class='ovr', average='weighted') #Sensitive to class imbalance even when average == 'macro'
print(f"roc_ovo_macro: {logistic_roc_ovo_macro}")
print(f"roc_ovr_weighted: {logistic_roc_ovr_weighted}")

# Show feature importance as a table
eli5.show_weights(logistic_reg, feature_names = cla_feature_names)


# 5.1.2.2. Support Vector Machine
# SVC
sv_cla = SVC(C=10, gamma=0.1, probability=True, kernel='rbf')
sv_cla.fit(x_train_cla, y_train_cla)
sv_cla_predictions = sv_cla.predict(x_test_cla)
sv_cla_results = classification_report(y_test_cla, sv_cla_predictions, zero_division=True, output_dict=True)

# Confusion matrix and Classification report
draw_confusion_matrix(sv_cla)
print(classification_report(y_test_cla, sv_cla_predictions, zero_division=True))

# balanced_accuracy
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score
sv_cla_balanced_accuracy = balanced_accuracy_score(y_test_cla, sv_cla_predictions)
print(f'balanced_accuracy: {sv_cla_balanced_accuracy}')

# ROC_AUC score
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
sv_cla_roc_ovo_macro = roc_auc_score(y_test_cla, sv_cla.predict_proba(x_test_cla), multi_class='ovo', average='macro') #Insensitive to class imbalance when average == 'macro'
sv_cla_roc_ovr_weighted = roc_auc_score(y_test_cla, sv_cla.predict_proba(x_test_cla), multi_class='ovr', average='weighted') #Sensitive to class imbalance even when average == 'macro'
print(f"roc_ovo_macro: {sv_cla_roc_ovo_macro}")
print(f"roc_ovr_weighted: {sv_cla_roc_ovr_weighted}")

# Show feature importance as a table
#eli5.show_weights(sv_cla, feature_names = cla_feature_names)


# 5.1.2.3. Decision Tree
# Decision Tree Classification
dt_cla = DecisionTreeClassifier(random_state=0)
dt_cla.fit(x_train_cla, y_train_cla)
dt_cla_predictions = dt_cla.predict(x_test_cla)
dt_cla_results = classification_report(y_test_cla, dt_cla_predictions, zero_division=True, output_dict=True)

# Confusion matrix and Classification report
draw_confusion_matrix(dt_cla)
print(classification_report(y_test_cla, dt_cla_predictions, zero_division=True))

# balanced_accuracy
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score
dt_cla_balanced_accuracy = balanced_accuracy_score(y_test_cla, dt_cla_predictions)
print(f'balanced_accuracy: {dt_cla_balanced_accuracy}')

# ROC_AUC score
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
dt_cla_roc_ovo_macro = roc_auc_score(y_test_cla, dt_cla.predict_proba(x_test_cla), multi_class='ovo', average='macro') #Insensitive to class imbalance when average == 'macro'
dt_cla_roc_ovr_weighted = roc_auc_score(y_test_cla, dt_cla.predict_proba(x_test_cla), multi_class='ovr', average='weighted') #Sensitive to class imbalance even when average == 'macro'
print(f"roc_ovo_macro: {dt_cla_roc_ovo_macro}")
print(f"roc_ovr_weighted: {dt_cla_roc_ovr_weighted}")

# Show feature importance as a table
eli5.show_weights(dt_cla, feature_names = cla_feature_names)


def visualize_decision_tree_feature_importance_cla():
    dt_cla_importances_df = pd.DataFrame(pd.Series(dt_cla.feature_importances_, index=X_cla.columns),
                                         columns=['Importance']).sort_values('Importance', ascending=False)
    # Visualize the feature importance of the trained tree
    plt.figure(figsize=(15, 10))
    missing_value_graph = sns.barplot(y=dt_cla_importances_df.index, x="Importance", data=dt_cla_importances_df,
                                      orient="h")
    missing_value_graph.set_title("Feature importance by Decision Tree Classification", fontsize=20)
    missing_value_graph.set_ylabel("Features")
    plt.show()


# Get the feature importance as a dataframe
visualize_decision_tree_feature_importance_cla()


# https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
dt_cla_graph = export_graphviz(dt_cla, out_file=None, max_depth=2, filled=True, feature_names=cla_feature_names)
graphviz.Source(dt_cla_graph)


def visualize_pdp_dt_classification():
    # 1-D pdp plot
    dt_cla_pdp_goals = pdp.pdp_isolate(model=dt_cla, dataset=x_test_cla, model_features=cla_feature_names,
                                       feature='Duration')
    # plot it
    pdp.pdp_plot(dt_cla_pdp_goals, 'Duration')
    plt.show()


visualize_pdp_dt_classification()


def visualize_2D_pdp_dt_classification():
    # 2D Partial Dependence Plots
    features_to_plot = ['Start_Lng', 'Start_Lat']
    dt_cla_pdp_2D = pdp.pdp_interact(model=dt_cla, dataset=x_test_cla, model_features=cla_feature_names,
                                     features=features_to_plot)
    pdp.pdp_interact_plot(pdp_interact_out=dt_cla_pdp_2D, feature_names=features_to_plot, plot_type='contour')
    plt.show()


visualize_2D_pdp_dt_classification()


# 5.1.2.4. Gradient Boost Tree
# Gradient Boosting Classification
gbt_cla = GradientBoostingClassifier(learning_rate=0.1, max_depth=10, min_impurity_decrease=0.1, min_samples_leaf=2, n_estimators=100, random_state=0)
gbt_cla.fit(x_train_cla, y_train_cla)
gbt_cla_predictions = gbt_cla.predict(x_test_cla)
gbt_cla_results = classification_report(y_test_cla, gbt_cla_predictions, zero_division=True, output_dict=True)

# Confusion matrix and Classification report
draw_confusion_matrix(gbt_cla)
print(classification_report(y_test_cla, gbt_cla_predictions, zero_division=True))

# balanced_accuracy
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score
gbt_cla_balanced_accuracy = balanced_accuracy_score(y_test_cla, gbt_cla_predictions)
print(f'balanced_accuracy: {gbt_cla_balanced_accuracy}')

# ROC_AUC score
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
gbt_cla_roc_ovo_macro = roc_auc_score(y_test_cla, gbt_cla.predict_proba(x_test_cla), multi_class='ovo', average='macro') #Insensitive to class imbalance when average == 'macro'
gbt_cla_roc_ovr_weighted = roc_auc_score(y_test_cla, gbt_cla.predict_proba(x_test_cla), multi_class='ovr', average='weighted') #Sensitive to class imbalance even when average == 'macro'
print(f"roc_ovo_macro: {gbt_cla_roc_ovo_macro}")
print(f"roc_ovr_weighted: {gbt_cla_roc_ovr_weighted}")
# Show feature importance as a table
eli5.show_weights(gbt_cla, feature_names = cla_feature_names)


def visualize_gbt_cla_feature_importance():
    gbt_cla_importances_std = np.std([tree[0].feature_importances_ for tree in gbt_cla.estimators_], axis=0)
    gbt_cla_importances = pd.Series(gbt_cla.feature_importances_, index=X_cla.columns)
    gbt_cla_importances_df = pd.DataFrame(gbt_cla_importances, columns=['Importance'])
    gbt_cla_importances_df['Std'] = gbt_cla_importances_std
    gbt_cla_importances_df.sort_values('Importance', ascending=True, inplace=True)
    fig, ax = plt.subplots(figsize=(15, 10))
    gbt_cla_importances_df['Importance'].plot.barh(xerr=gbt_cla_importances_df['Std'], color='cornflowerblue', ax=ax)
    ax.set_title("Feature importances using MDI of Gradient Boosting Classification", fontsize=22)
    ax.set_xlabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()


# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
# Calculate Standard Deviation of each feature for all the trees
visualize_gbt_cla_feature_importance()


# 5.1.2.5. Random Forest
# Random Forest Classification
rf_cla = RandomForestClassifier(random_state=0)
rf_cla.fit(x_train_cla, y_train_cla)
rf_cla_predictions = rf_cla.predict(x_test_cla)
rf_cla_results = classification_report(y_test_cla, rf_cla_predictions, zero_division=True, output_dict=True)

# Confusion matrix and Classification report
draw_confusion_matrix(rf_cla)
print(classification_report(y_test_cla, rf_cla_predictions, zero_division=True))

# balanced_accuracy
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score
rf_cla_balanced_accuracy = balanced_accuracy_score(y_test_cla, rf_cla_predictions)
print(f'balanced_accuracy: {rf_cla_balanced_accuracy}')

# ROC_AUC score
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
rf_cla_roc_ovo_macro = roc_auc_score(y_test_cla, rf_cla.predict_proba(x_test_cla), multi_class='ovo', average='macro') #Insensitive to class imbalance when average == 'macro'
rf_cla_roc_ovr_weighted = roc_auc_score(y_test_cla, rf_cla.predict_proba(x_test_cla), multi_class='ovr', average='weighted') #Sensitive to class imbalance even when average == 'macro'
print(f"roc_ovo_macro: {rf_cla_roc_ovo_macro}")
print(f"roc_ovr_weighted: {rf_cla_roc_ovr_weighted}")
# Show feature importance as a table
eli5.show_weights(rf_cla, feature_names = cla_feature_names)


def visualize_rf_cla_feature_importance():
    rf_cla_importances_std = np.std([tree.feature_importances_ for tree in rf_cla.estimators_], axis=0)
    rf_cla_importances = pd.Series(rf_cla.feature_importances_, index=X_cla.columns)
    rf_cla_importances_df = pd.DataFrame(rf_cla_importances, columns=['Importance'])
    rf_cla_importances_df['Std'] = rf_cla_importances_std
    rf_cla_importances_df.sort_values('Importance', ascending=True, inplace=True)
    fig, ax = plt.subplots(figsize=(15, 10))
    rf_cla_importances_df['Importance'].plot.barh(xerr=rf_cla_importances_df['Std'], color='cornflowerblue', ax=ax)
    ax.set_title("Feature importances using MDI of Random Forest Classification", fontsize=22)
    ax.set_xlabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()


# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
# Calculate Standard Deviation of each feature for all the trees
visualize_rf_cla_feature_importance()


# 5.1.2.6. XGBoost
# XGB Classification
# https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
xgb_cla = XGBClassifier(learning_rate=0.3, max_depth=20, n_estimators=100, eval_metric='mlogloss', random_state=0)
xgb_cla.fit(x_train_cla, y_train_cla)
xgb_cla_predictions = xgb_cla.predict(x_test_cla)
xgb_cla_results = classification_report(y_test_cla, xgb_cla_predictions, zero_division=True, output_dict=True)

# Confusion matrix and Classification report
draw_confusion_matrix(xgb_cla)
print(classification_report(y_test_cla, xgb_cla_predictions, zero_division=True))

# balanced_accuracy
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score
xgb_cla_balanced_accuracy = balanced_accuracy_score(y_test_cla, xgb_cla_predictions)
print(f'balanced_accuracy: {xgb_cla_balanced_accuracy}')

# ROC_AUC score
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
xgb_cla_roc_ovo_macro = roc_auc_score(y_test_cla, xgb_cla.predict_proba(x_test_cla), multi_class='ovo', average='macro') #Insensitive to class imbalance when average == 'macro'
xgb_cla_roc_ovr_weighted = roc_auc_score(y_test_cla, xgb_cla.predict_proba(x_test_cla), multi_class='ovr', average='weighted') #Sensitive to class imbalance even when average == 'macro'
print(f"roc_ovo_macro: {xgb_cla_roc_ovo_macro}")
print(f"roc_ovr_weighted: {xgb_cla_roc_ovr_weighted}")
# Show feature importance as a table
eli5.show_weights(xgb_cla, feature_names = cla_feature_names)


def visualize_xgb_cla_feature_importance():
    xgb_cla_importances_df = pd.DataFrame(pd.Series(xgb_cla.feature_importances_, index=X_cla.columns),
                                          columns=['Importance']).sort_values('Importance', ascending=False)
    # Visualize the feature importance of the trained tree
    plt.figure(figsize=(15, 10))
    missing_value_graph = sns.barplot(y=xgb_cla_importances_df.index, x="Importance", data=xgb_cla_importances_df,
                                      orient="h")
    missing_value_graph.set_title("Feature importance by XGB Classification", fontsize=20)
    missing_value_graph.set_ylabel("Features")
    plt.show()


# Get the feature importance as a dataframe
visualize_xgb_cla_feature_importance()


def visualize_xgb_cla_pdp_crossing():
    # 1-D pdp plot
    xgb_cla_pdp_goals = pdp.pdp_isolate(model=xgb_cla, dataset=x_test_cla, model_features=cla_feature_names,
                                        feature='Crossing')
    # plot it
    pdp.pdp_plot(xgb_cla_pdp_goals, 'Crossing')
    plt.show()


visualize_xgb_cla_pdp_crossing()


def visualize_xgb_cla_pdp_2d():
    # 2D Partial Dependence Plots
    features_to_plot = ['Side', 'Duration']
    xgb_cla_pdp_2D = pdp.pdp_interact(model=xgb_cla, dataset=x_test_cla, model_features=cla_feature_names,
                                      features=features_to_plot)
    pdp.pdp_interact_plot(pdp_interact_out=xgb_cla_pdp_2D, feature_names=features_to_plot, plot_type='contour')
    plt.show()


visualize_xgb_cla_pdp_2d()

# 5.1.2.7. Multi-layer Perceptron Classification
# Multi-layer Perceptron classifier.
mlpc_cla = MLPClassifier(activation='tanh', hidden_layer_sizes=(100, 100), learning_rate='invscaling', random_state = 0)
mlpc_cla.fit(x_train_cla, y_train_cla)
mlpc_cla_predictions = mlpc_cla.predict(x_test_cla)
mlpc_cla_results = classification_report(y_test_cla, mlpc_cla_predictions, zero_division=True, output_dict=True)

# Confusion matrix and Classification report
draw_confusion_matrix(mlpc_cla)
print(classification_report(y_test_cla, mlpc_cla_predictions, zero_division=True))

# balanced_accuracy
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score
mlpc_cla_balanced_accuracy = balanced_accuracy_score(y_test_cla, mlpc_cla_predictions)
print(f'balanced_accuracy: {mlpc_cla_balanced_accuracy}')

# ROC_AUC score
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
mlpc_cla_roc_ovo_macro = roc_auc_score(y_test_cla, mlpc_cla.predict_proba(x_test_cla), multi_class='ovo', average='macro') #Insensitive to class imbalance when average == 'macro'
mlpc_cla_roc_ovr_weighted = roc_auc_score(y_test_cla, mlpc_cla.predict_proba(x_test_cla), multi_class='ovr', average='weighted') #Sensitive to class imbalance even when average == 'macro'
print(f"roc_ovo_macro: {mlpc_cla_roc_ovo_macro}")
print(f"roc_ovr_weighted: {mlpc_cla_roc_ovr_weighted}")



# 5.1.2.8. Model Comparison
# Gather all the classification performance in one table
cla_results = pd.DataFrame([
    (logistic_balanced_accuracy, logistic_reg_results['accuracy'], logistic_reg_results['weighted avg']['precision'], logistic_reg_results['weighted avg']['recall'], logistic_reg_results['weighted avg']['f1-score'], logistic_roc_ovo_macro, logistic_roc_ovr_weighted),
    (sv_cla_balanced_accuracy, sv_cla_results['accuracy'], sv_cla_results['weighted avg']['precision'], sv_cla_results['weighted avg']['recall'], sv_cla_results['weighted avg']['f1-score'], sv_cla_roc_ovo_macro, sv_cla_roc_ovr_weighted),
    (dt_cla_balanced_accuracy, dt_cla_results['accuracy'], dt_cla_results['weighted avg']['precision'], dt_cla_results['weighted avg']['recall'], dt_cla_results['weighted avg']['f1-score'], dt_cla_roc_ovo_macro, dt_cla_roc_ovr_weighted),
    (gbt_cla_balanced_accuracy, gbt_cla_results['accuracy'], gbt_cla_results['weighted avg']['precision'], gbt_cla_results['weighted avg']['recall'], gbt_cla_results['weighted avg']['f1-score'], gbt_cla_roc_ovo_macro, gbt_cla_roc_ovr_weighted),
    (rf_cla_balanced_accuracy, rf_cla_results['accuracy'], rf_cla_results['weighted avg']['precision'], rf_cla_results['weighted avg']['recall'], rf_cla_results['weighted avg']['f1-score'], rf_cla_roc_ovo_macro, rf_cla_roc_ovr_weighted),
    (xgb_cla_balanced_accuracy, xgb_cla_results['accuracy'], xgb_cla_results['weighted avg']['precision'], xgb_cla_results['weighted avg']['recall'], xgb_cla_results['weighted avg']['f1-score'], xgb_cla_roc_ovo_macro, xgb_cla_roc_ovr_weighted),
    (mlpc_cla_balanced_accuracy, mlpc_cla_results['accuracy'], mlpc_cla_results['weighted avg']['precision'], mlpc_cla_results['weighted avg']['recall'], mlpc_cla_results['weighted avg']['f1-score'], mlpc_cla_roc_ovo_macro, mlpc_cla_roc_ovr_weighted)],
    columns=['Accuracy(Balanced)', 'Accuracy','Precision(Weighted_avg)', 'Recall(Weighted_avg)', 'F1-score(Weighted_avg)', 'Roc_ovo(macro)', 'Roc_ovr(weighted)'],
    index= ['Logistics Regression',
            'Support Vector Machine',
            'Decision Tree',
            'Gradient Boosting Tree',
            'Random Forest',
            'XGBoost',
            'Multi-layer Perceptron'])

cla_results.sort_values(by=['F1-score(Weighted_avg)'], ascending=False)


# 5.1.2.9. Deal with Imbalanced data
# Form the train data for sampling
train_cla_df = pd.concat([x_train_cla, y_train_cla], axis=1)

# Over-sampling and Under-sampling
size_l = len(train_cla_df[train_cla_df["Severity"]==2].index)
size_s = len(train_cla_df[train_cla_df["Severity"]==1].index)

train_cla_over = pd.DataFrame()
train_cla_under = pd.DataFrame()

for i in range(1,5):
    class_df = train_cla_df[train_cla_df["Severity"]==i]
    train_cla_over = train_cla_over.append(class_df.sample(size_l, random_state=1, replace=True))
    train_cla_under = train_cla_under.append(class_df.sample(size_s, random_state=1, replace=False))

print(f'Over-sampling: \n{train_cla_over.Severity.value_counts()}')
print(f'Under-sampling: \n{train_cla_under.Severity.value_counts()}')


# Try on over-sampling data
# XGB Classification
xgb_cla = XGBClassifier(learning_rate=0.3, max_depth=20, n_estimators=100, eval_metric='mlogloss', random_state=0)
xgb_cla.fit(train_cla_over.drop('Severity', axis=1), train_cla_over['Severity'])
xgb_cla_predictions_over = xgb_cla.predict(x_test_cla)

# Confusion matrix and Classification report
draw_confusion_matrix(xgb_cla)
print(classification_report(y_test_cla, xgb_cla_predictions_over, zero_division=True))



# Try on under-sampling data
# XGB Classification
xgb_cla = XGBClassifier(learning_rate=0.3, max_depth=20, n_estimators=100, eval_metric='mlogloss', random_state=0)
xgb_cla.fit(train_cla_under.drop('Severity', axis=1), train_cla_under['Severity'])
xgb_cla_predictions_under = xgb_cla.predict(x_test_cla)

# Confusion matrix and Classification report
draw_confusion_matrix(xgb_cla)
print(classification_report(y_test_cla, xgb_cla_predictions_under, zero_division=True))


# 5.1.2.10. Severity Prediction Visualization
# Orlando latitude and longitude values
# https://www.latlong.net/place/orlando-fl-usa-1947.html
orlando_lat = 28.538336
orlando_long = -81.379234

# Generate a map of Orlando
orlando_map = folium.Map(location=[orlando_lat, orlando_long], zoom_start=12)

# Instantiate a mark cluster object for the incidents in the dataframe
accidents = folium.plugins.MarkerCluster().add_to(orlando_map)

# Loop through the dataframe and add each data point to the mark cluster
for lat, lng, label in zip(x_test_cla['Start_Lat'], x_test_cla['Start_Lng'], xgb_cla_predictions.astype(str)):
    if label == '4':
        folium.Marker(
            location=[lat, lng],
            icon=folium.Icon(color="red", icon="warning-sign"), #https://getbootstrap.com/docs/3.3/components/
            popup=label,
            ).add_to(accidents)
    elif label == '3':
        folium.Marker(
            location=[lat, lng],
            icon=folium.Icon(color="lightred", icon="warning-sign"),
            popup=label,
            ).add_to(accidents)
    elif label == '2':
        folium.Marker(
            location=[lat, lng],
            icon=folium.Icon(color="orange", icon="warning-sign"),
            popup=label,
            ).add_to(accidents)
    elif label == '1':
        folium.Marker(
            location=[lat, lng],
            icon=folium.Icon(color="beige", icon="warning-sign"),
            popup=label,
            ).add_to(accidents)
# Display map
orlando_map


# 5.2. End Time Prediction
# 5.2.1 Parameter Turning
######Stop line######
# The following code can not run out at Kaggle because of the resource limitation; Have already done it on Google Colab. Therefore, stop here.

# Linear Regression
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
lr_pipe = make_pipeline(LinearRegression())
# lr_pipe.get_params().keys()
lr_param = {
    'linearregression__fit_intercept': [True, False],
    'linearregression__normalize': [True, False]
}

lr_search = GridSearchCV(lr_pipe,
                         lr_param,
                         scoring="neg_root_mean_squared_error",
                         n_jobs=-1,
                         cv = 5)
lr_search.fit(X_reg, Y_reg)
print(f'Best Params: {lr_search.best_params_} \nBest score: {-(lr_search.best_score_)}')


# SVR
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
# https://7125messi.github.io/post/svm%E8%B0%83%E4%BC%98%E8%AF%A6%E8%A7%A3/
svr_pipe = make_pipeline(StandardScaler(), SVR())
svr_param = {
    'svr__C': [0.1, 1.0, 10],
    'svr__gamma': [1, 0.1, 0.01]
}

svr_search = GridSearchCV(svr_pipe,
                         svr_param,
                         scoring="neg_root_mean_squared_error",
                         n_jobs=-1,
                         cv = 5)
svr_search.fit(X_reg, Y_reg)
print(f'Best Params: {svr_search.best_params_} \nBest score: {-(svr_search.best_score_)}')

# Decision Tree Regression
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
dtr_pipe = make_pipeline(DecisionTreeRegressor(random_state=0))
dtr_param = {
    'decisiontreeregressor__max_depth': [5, 10, 20],
    'decisiontreeregressor__min_samples_leaf': [2, 5, 10],
    'decisiontreeregressor__min_impurity_decrease': [0.1, 0.2, 0.5]

}

dtr_search = GridSearchCV(dtr_pipe,
                          dtr_param,
                          scoring="neg_root_mean_squared_error",
                          n_jobs=-1,
                          cv=5)
dtr_search.fit(X_reg, Y_reg)
print(f'Best Params: {dtr_search.best_params_} \nBest score: {-(dtr_search.best_score_)}')

# Gradient Boosting Regression
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
gbtr_pipe = make_pipeline(GradientBoostingRegressor(random_state=0))
gbtr_param = {
    'gradientboostingregressor__learning_rate': [0.1, 0.3, 0.7],
    'gradientboostingregressor__max_depth': [5, 10, 20],
    'gradientboostingregressor__n_estimators': [50, 100, 200],
    'gradientboostingregressor__min_impurity_decrease': [0.1, 0.2, 0.5],
    'gradientboostingregressor__min_samples_leaf': [2, 5, 10]

}

gbtr_search = GridSearchCV(gbtr_pipe,
                           gbtr_param,
                           scoring="neg_root_mean_squared_error",
                           n_jobs=-1,
                           cv=5)
gbtr_search.fit(X_reg, Y_reg)
print(f'Best Params: {gbtr_search.best_params_} \nBest score: {-(gbtr_search.best_score_)}')

# Random Forest Regression
rfr_pipe = make_pipeline(RandomForestRegressor(random_state=0))
rfr_param = {
    'randomforestregressor__max_depth': [5, 10, 20],
    'randomforestregressor__n_estimators': [50, 100, 200],
    'randomforestregressor__min_impurity_decrease': [0.1, 0.2, 0.5],
    'randomforestregressor__min_samples_leaf': [2, 5, 10],

}

rfr_search = GridSearchCV(rfr_pipe,
                          rfr_param,
                          scoring="neg_root_mean_squared_error",
                          n_jobs=-1,
                          cv=5)
rfr_search.fit(X_reg, Y_reg)
print(f'Best Params: {rfr_search.best_params_} \nBest score: {-(rfr_search.best_score_)}')


# XGB Regression
xgbr_pipe = make_pipeline(XGBRegressor(random_state=0))
xgbr_param = {
              'xgbregressor__learning_rate': [0.1, 0.3, 0.7],
              'xgbregressor__max_depth': [30, 50, 100],
              'xgbregressor__n_estimators': [50, 100, 200]
             }

xgbr_search = GridSearchCV(xgbr_pipe,
                         xgbr_param,
                         scoring="neg_root_mean_squared_error",
                         n_jobs=-1,
                         cv = 5)
xgbr_search.fit(X_reg, Y_reg)
print(f'Best Params: {xgbr_search.best_params_} \nBest score: {-(xgbr_search.best_score_)}')

# Multi-layer Perceptron regressor
mlpr_pipe = make_pipeline(MLPRegressor(random_state=0))
mlpr_param = {
              'mlpregressor__activation': ['logistic', 'tanh', 'relu'],
              'mlpregressor__hidden_layer_sizes': [(100,), (50, 100), (50, 75, 100)],
              'mlpregressor__learning_rate': ['invscaling', 'adaptive']
             }

mlpr_search = GridSearchCV(mlpr_pipe,
                         mlpr_param,
                         scoring="neg_root_mean_squared_error",
                         n_jobs=-1,
                         cv = 5)
mlpr_search.fit(X_reg, Y_reg)
print(f'Best Params: {mlpr_search.best_params_} \nBest score: {-(mlpr_search.best_score_)}')


# 5.2.2 Model Comparison
pd.DataFrame([-(lr_search.best_score_), -(svr_search.best_score_), -(dtr_search.best_score_), -(gbtr_search.best_score_), -(rfr_search.best_score_), -(xgbr_search.best_score_), -(mlpr_search.best_score_)],
             columns=['RMSE'],
             index= ['Linear Regression',
                    'Support Vector Machine',
                    'Decision Tree',
                    'Gradient Boosting Tree',
                    'Random Forest',
                    'XGBoost',
                    'Multi-layer Perceptron']).sort_values(by=['RMSE'])

# 5.2.3 Visualization

