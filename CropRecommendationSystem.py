import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df_ = pd.read_csv("Crop_recommendation.csv")
df = df_.copy()


print(df.head(5))


"""
The dataset includes the following features:

N: Nitrogen content in the soil
P: Phosphorus content in the soil
K: Potassium content in the soil
temperature: Temperature in degrees Celsius
humidity: Relative humidity in percentage
ph: pH value of the soil
rainfall: Rainfall in mm
label: Crop type (target variable)
"""


# EDA (EXPLORATORY DATA ANALYSIS)
# Check dataframe

def check_df(dataframe, head=5):
    print('##################### Shape #####################')
    print(dataframe.shape)
    print('##################### Types #####################')
    print(dataframe.dtypes)
    print('##################### Head #####################')
    print(dataframe.head(head))
    print('##################### Tail #####################')
    print(dataframe.tail(head))
    print('##################### NA #####################')
    print(dataframe.isnull().sum())
    print('##################### Quantiles #####################')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

# Unique labels, and how many?

print(df["label"].nunique())  # 22
print(df["label"].unique())
print(df["label"].value_counts())  # 100 for all


# To check the types of the columns

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != 'O']
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == 'O']
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    num_cols = [col for col in num_cols if col not in num_but_cat]
    print(f'Observations: {dataframe.shape[0]}')
    print(f'Variables: {dataframe.shape[1]}')
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat


cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

print("cat_cols= ", cat_cols)
print("num_cols= ", num_cols)
print("cat_but_car= ", cat_but_car)
print("num_but_car= ", num_but_cat)


var = df[num_cols].describe().T
print(var)  # --> seems like no outliers looking at the distribution. Can be for P or K looking at the max value.


# Check missing values

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1,
                           keys=['number of missing values', 'percentage of missing values'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

na_cols = missing_values_table(df, True)  # --> No missing values as also seen in Check Df function


# Check for outliers
# As df.shape is (2200, 8), I only have 2200 data points, so I will be choosing q1=0.05,
# q3=0.95; instead of q1=0.25, q3=0.75.

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit


for elem in num_cols:
    print(elem, outlier_thresholds(df, elem))

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for elem in num_cols:
    print(elem, check_outlier(df, elem))  # --> No outliers


# Multivariate Outlier Analysis  ( Local Outlier Factor )

new_df = pd.get_dummies(df, columns=['label'])
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(new_df)

df_scores = clf.negative_outlier_factor_
print(df_scores[0:5])
print(np.sort(df_scores)[:])

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()
# 10. index selected.
th = np.sort(df_scores)[10]  # 10 outliers: 408, 410, 445, 480, 493, 1724, 1731, 1746, 1852, 1863
#  --> drop the 10 outliers
df.drop(df[df_scores < th].index, inplace=True)
df.reset_index()
print(df.shape)  # (2190, 12)


# The distribution of our numeric variables

def num_summary(dataframe, num_cols, bins=20):
    for numerical_col in num_cols:
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        dataframe[numerical_col].hist(bins=bins)
        plt.xlabel(numerical_col)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {numerical_col}')

        plt.subplot(1, 2, 2)
        dataframe[numerical_col].plot(kind='box')
        plt.xlabel(numerical_col)
        plt.title(f'Boxplot of {numerical_col}')

        plt.tight_layout()
        plt.show()


num_summary(df, num_cols)

# To see the averages of columns according to each crop:

grouped = df.groupby(by='label').mean().reset_index()

# TOP 3 Most ... requiring crops:

print(f'---------')
for i in grouped.columns[1:]:
    print(f'Top 3 Most {i} requiring crops:')
    print(f'--------')
    for j, k in grouped.sort_values(by=i, ascending=False)[:3][['label', i]].values:
        print(f'{j} --> {k}')
    print(f'--------')

# TOP 3 Least ... requiring crops:

print(f'--------')
for i in grouped.columns[1:]:
    print(f'Top 3 Least {i} requiring crops:')
    print(f'--------')
    for j, k in grouped.sort_values(by=i)[:3][['label', i]].values:
        print(f'{j} --> {k}')
    print(f'--------')

# Correlation analysis:

# Create a copy of the DataFrame without the 'label' column
df_corr = df.drop(columns=['label'])

# Correlation analysis
figure = plt.figure(figsize=(12, 6))
sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Scatter Plot of K and P --> K and P has + 0.74 a strong positive correlation
plt.scatter(df_corr['K'], df_corr['P'], color='blue', marker='o')
plt.xlabel('K')
plt.ylabel('P')
plt.title('Scatter Plot of K vs P')
plt.grid(True)
plt.show()


# PREPROCESSING.
# Feature Engineering

indices = np.arange(len(df))
indices_train, indices_test = train_test_split(indices, test_size=0.2, random_state=0)
indices_train, indices_val = train_test_split(indices_train, test_size=0.2, random_state=0)

percentiles = np.arange(0.05, 1.01, 0.05)
print(df["temperature"].describe(percentiles).T)

df["NEW_temperature"] = pd.cut(df["temperature"],
                               bins=[-np.inf, 10, 20, 25, 30, 35, np.inf],
                               labels=['very_cold', 'cold', 'moderate', 'moderate_hot', 'hot', 'very_hot'])

df["NEW_P_K"] = df["P"] * df["K"]

df["NEW_ph"] = pd.cut(df["ph"],
                      bins=[-np.inf, 7, 8, np.inf],
                      labels=['acidic', 'neutral', 'alkaline'])

percentiles = np.arange(0.05, 1.01, 0.05)
print(df["rainfall"].describe(percentiles).T)

df["NEW_rainfall"] = pd.cut(df["rainfall"],
                            bins=[-np.inf, 50, 90, 115, 190, np.inf],
                            labels=['very_low_rain', 'low_rain', 'moderate_rain', 'heavy_rain', 'very_heavy_rain'])

df.head()


# Again grab col names with new columns:

def grab_col_names(df, target):
    df = df.drop(columns=[target], errors='ignore')
    cat_cols = [col for col in df.columns if df[col].dtype == 'O' or df[col].dtype == 'category']
    num_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    cat_but_car = [col for col in cat_cols if len(df[col].unique()) > 20]
    num_but_cat = [col for col in num_cols if len(df[col].unique()) < 10]
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car, num_but_cat


cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, target="label")

print("cat_cols= ", cat_cols)
print("num_cols= ", num_cols)
print("cat_but_car= ", cat_but_car)
print("num_but_car= ", num_but_cat)


# cat_cols=  ['NEW_temperature', 'NEW_ph', 'NEW_rainfall'] --> I now have categorical columns
# num_cols=  ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'NEW_P_K']
# cat_but_car=  []
# num_but_car=  []


# Analyzing categorical variables
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


cat_summary(df, 'NEW_temperature', plot=True)
cat_summary(df, 'NEW_ph', plot=True)
cat_summary(df, 'NEW_rainfall', plot=True)


# Dummy encoding

def dummy_encode(df, column):
    dummy_df = pd.get_dummies(df[column], prefix=column, drop_first=True)
    dummy_df = dummy_df.astype(int)
    df = pd.concat([df, dummy_df], axis=1)
    return df


for var in cat_cols:
    df = dummy_encode(df, var)

print(df.head())  # To control


# Function to standardize
def standardize(df, var, train_ind):
    var_train = df[var].iloc[train_ind]
    var_train_mean = np.mean(var_train)
    var_train_std = np.std(var_train)
    df.loc[:, var] = (df[var] - var_train_mean) / var_train_std
    return df


# Apply standardization to the DataFrame
for var in num_cols:
    df = standardize(df, var, indices_train)


# Label encoding
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])


# Drop encoded columns
columns_to_drop = ['NEW_temperature', 'NEW_ph', 'NEW_rainfall']
df = df.drop(columns_to_drop, axis=1)


# To control
print(df.head())


# Split the DataFrame into training, validation, and test sets
train_df = df.iloc[indices_train].reset_index(drop=True)
test_df = df.iloc[indices_test].reset_index(drop=True)
val_df = df.iloc[indices_val].reset_index(drop=True)


# Y, X SPLIT
y_train = train_df["label"]
X_train = train_df.drop(["label"], axis=1)
X_train = np.array(X_train)


y_val = val_df["label"]
X_val = val_df.drop(["label"], axis=1)
X_val = np.array(X_val)


y_test = test_df["label"]
X_test = test_df.drop(["label"], axis=1)
X_test = np.array(X_test)


def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 4)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Test Accuracy Score: {0}'.format(acc), size=10)
    plt.show()


print("Base Models....")


models = [('LR', LogisticRegression(max_iter=1000)),
          ('KNN', KNeighborsClassifier(metric='euclidean')),
          ("SVC", SVC(probability=True)),
          ("CART", DecisionTreeClassifier()),
          ("RF", RandomForestClassifier()),
          ('GBM', GradientBoostingClassifier()),
          ('XGBoost', XGBClassifier(eval_metric='mlogloss'))]


# Define parameter grids for each model
param_grids = {
    'LR': {
        'model': LogisticRegression(max_iter=1000),
        'params': {
            'solver': ['lbfgs', 'liblinear'],
            'C': [0.01, 0.1, 1.0, 10]
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(metric='euclidean'),
        'params': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    },
    'SVC': {
        'model': SVC(probability=True),
        'params': {
            'C': [0.01, 0.1, 1.0, 10],
            'kernel': ['linear', 'rbf', 'poly']
        }
    },
    'CART': {
        'model': DecisionTreeClassifier(),
        'params': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    },
    'RF': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_features': ['sqrt', 'log2', None]
        }
    },
    'GBM': {
        'model': GradientBoostingClassifier(),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(eval_metric='mlogloss'),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }
}


for name, model_info in param_grids.items():
    print(f"########## Tuning {name} ##########")
    grid_search = GridSearchCV(estimator=model_info['model'], param_grid=model_info['params'], cv=5, scoring='accuracy',
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best Parameters for {name}: {grid_search.best_params_}")

    y_pred = best_model.predict(X_val)

    plot_confusion_matrix(y_val, y_pred)

    scoring = ["accuracy", "f1_macro", "roc_auc_ovr", "precision_macro", "recall_macro"]
    cv_results = cross_validate(best_model, X_train, y_train, cv=10, scoring=scoring)
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"AUC: {round(cv_results['test_roc_auc_ovr'].mean(), 4)}")
    print(f"Macro-Recall: {round(cv_results['test_recall_macro'].mean(), 4)}")
    print(f"Macro-Precision: {round(cv_results['test_precision_macro'].mean(), 4)}")
    print(f"Macro-F1: {round(cv_results['test_f1_macro'].mean(), 4)}")


# Finally, evaluate the best model on the test set
best_model.fit(X_train, y_train)
y_pred_test = best_model.predict(X_test)
plot_confusion_matrix(y_test, y_pred_test)


"""
--> The Random Forest Classifier (RF) stands out as the best model
overall based on the combination of accuracy, AUC, macro-recall, macro-precision,
and macro-F1 score. It provides the highest performance in most metrics
"""