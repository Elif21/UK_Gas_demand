import pandas as pd
import seaborn as sns; sns.set(color_codes=True)
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from scipy.stats import pearsonr
from scipy import stats
from sklearn import linear_model
from sklearn.utils import shuffle
import sklearn
from sklearn import preprocessing
import pickle
import pandas as pd


df=pd.read_csv('C:/Users/Michel/Documents/Data Scientist sessions/UK_GAS PROJECT/Notebook/uk gas demand.csv')
#df=pd.read_csv("E:/Battery Project/Data/UK Gas Demand/uk gas demand.csv")
df.info()
df.shape
#pd.close()

#drop_cols = ['stginjection', 'supply', 'supplyf', 't', 'dateid01', 'consf', 'consldz', 'consldzf', 'consnldz', 'consnldzf', 'demanddown', 'demandup', 'dfri', 'dindex', 'dintexport', 'dintimport', 'dmon', 'dsat', 'dsun', 'dthu', 'dtue', 'dwed', 'dweek', 'dweekend', 'ginject', 'ginjectf', 'gwdrawal', 'gwdrawalb', 'gwdrawalf', 'intcon', 'intconf', 'irelandx', 'mapr', 'maug', 'mdec', 'mfeb', 'mjan', 'mjul', 'mjun', 'mmar', 'mmay', 'mnov', 'moct', 'msep', 'netacc', 'netstorage', 'netstoragef', 'phigh', 'plow', 'price_sap', 'pricef', 'tempday1', 'tempde', 'tempn', 'teodemandf', 'week', 'wind',  'windc', 'windcn1', 'wk', 'wknd']
#df = df.drop(drop_2wcols, axis=1)

df.info()
df.tail() #this shows the last 5 observations of the data
df.shape

#this part counts the unique values.
print("Unique Value Count:")
cols = df.columns.tolist()
for col in cols:
  print(col + " = " + str(len(df[col].unique())))

df.describe()  # gives the statistics on the data

df_sort = df.sort_values(by='windcn', ascending=False)

#replace space in column name
df.columns = df.columns.str.replace(' ','')


#count by catagory - cross tabulate
make_dist = df.groupby('windcn').size ()

#Distribution of Categorical variable
make_dist.plot(title='Make Distribution')

#select all numerical variables
df_num = df.select_dtypes(include=['float', 'int64'])
df_num.head()

# Creating histograms
df_num.hist(bins=20)


#Correlation with the variable of interest can be used in time series (correlation between two time series, x and y),slop down corrolation is negative and up is positive
x=df['teodemand']
y=df['windcn']

#creating correlation matrix by using pearson method.
df_corr = df.corr(method='pearson')# they do not have same length


#Correlation plots using'pairplot'
for i in range(0, len(df_num.columns),1):
    sns.pairplot(df_num, y_vars=["windcn"],
                 x_vars=df_num.columns[i:i+1])

#plotting significant correlation in one plot (heatmap), to find the corrolation between independent(it is greater than 0.5 it is significant correlation, corrolation for target variable the highter the better the variable, high corrolation for independent feature is not good)
corr = df_num.drop('windcn',axis=1).corr()

# Plotting correlation between day temperature and day demand
plt.scatter(df['teodemand'], df['tempday'])

sns.pairplot(df)
plt.show()

corr=df.corr()
corr.style.background_gradient(cmap='coolwarm')

print("colmns with missing values: ")
print(df.columns[df.isnull().any()].tolist())

print(df.isnull().sum())

sns.catplot(x="windcn", y="teodemand", kind="box", data=df)

sns.catplot(x="tempday", y="teodemand", kind="box", data=df)

#box-plot (categorical variable)
box1 = sns.boxplot(x='teodemand', y='windcn', data=df)

box2 = sns.boxplot(x='Origin', y='windcn', data=df)

box3 = sns.boxplot(x='tempday', y='windcn', data=df)
tips=sns.load_dataset('tips')
ax=sns.regplot(x='windcn', y='tempday', data=df)

#regression plot
x = df["tempday"]
y = df["teodemand"]

slope, intercept, r, p, std_err = stats.linregress(x, y)
print(r)
print("slope:%f intercenpt: %f " % (slope, intercept) )
plt.plot(x, y, '+', label='original data')
plt.plot(x, intercept+slope*x, 'r', label='fitted line' )

#In order to train the model we need to create dummy model, why?

# Now we can use linear to predict grades like before

# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
best = 0
#x1=df.normalize()
#x1=preprocessing.normalize(x, norm='l1')
#x1=x.reshape(-1, 1)
x = df[['tempday', 'windcn']] #returns a numpy array
x_norm=(x-x.min())/(x.max()-x.min())
y=df[['teodemand']]
y_norm=(y-y.min())/(y.max()-y.min())
#  x_norm=x.reshape(1, -1)
#  X_l2 = preprocessing.normalize(x_norm, norm='l2')
best=0
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_norm, y_norm, test_size=0.20)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))
    # If the current model has a better score than one we've already trained then save it
    if acc > best:
        best = acc
        with open("df.pickle", "wb") as f:
            pickle.dump(linear, f)

# Using support vector machine
# This part part estimate and predict gas demand with respect to day temperature and wind chill factors.
x = df[['tempday', 'windcn']] #returns a numpy array
y=df[['teodemand']]
best=0
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_norm, y_norm, test_size=0.20)
    linear = linear_model.Ridge()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))
    # If the current model has a better score than one we've already trained then save it
    if acc > best:
        best = acc
        with open("df.pickle", "wb") as f:
            pickle.dump(linear, f)
