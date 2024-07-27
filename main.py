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
