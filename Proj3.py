# for dataframes and math
import pandas as pd
import numpy as np
import math

# for plotting 
import matplotlib.pyplot as plt
import seaborn as sns

# For our regression
import statsmodels.formula.api as smf

# reading the data in
url = "https://github.com/ArieBeresteanu/Econ1193_Spring2024/raw/main/cardata2005.csv"
cars = pd.read_csv(url)

cars.head()

# getting basic information about our data
nrows, ncols = cars.shape
print(f"There are {nrows} rows/cars and {ncols} columns.", "\n \n Data Frame Info:")
print(cars.info())

# using a lambda function to map 0,2,3, and 4 as numerical categories for cars
cars['category'] = cars['segm1'].map(lambda x: math.floor((x)/10)) 

# using a dictionary to specify the categories

categoryDict = {
    '0': 'passenger cars',
    '2': 'minivans',
    '3': 'SUV',
    '4': 'light trucks'   
}

# creating a column category name to descibe the type of car 
cars['categoryName'] = cars['category'].map(lambda x: categoryDict[str(x)])

# creating new columns to combine length and width, and to combine city and highway mpg
cars['mpg_combined'] = cars['mpg_city']*0.55+cars['mpg_highway']*0.45
cars['footprint'] = cars['width'] * cars['length'] /1000  #rescaling

# creating a summary statistics table
stats_table = cars.describe()
# dropping irrelevant columns
stats_table.drop(columns = ['year', 'firm_id', 'segm1'], inplace = True)
# transposing for clarity
stats_table.transpose()

# creating a small dataframe to show how many cars are hybrid
hybrid_counts = pd.DataFrame(cars['hybrid'].value_counts())

# for more clarity on what is hybrid and what is not
categoryDict2 = {
   '0': 'Not Hybrid',
   '1': 'Hybrid',
}
hybrid_counts['label'] = hybrid_counts.index.map(lambda x: categoryDict2[str(x)])
hybrid_counts

top_cars = cars[['Quantity', 'model']].sort_values(by = 'Quantity', ascending = False)
top_cars.head(3)

# we can just use .tail() to find the bottom three selling cars 
top_cars.tail(3)

nHH2005 = 113343000 # from the excel file on professor Beresteanu's Github
totalQuanity = sum(cars["Quantity"]) # to find number of cars sold
S0 = 1 - totalQuanity/nHH2005 # proportion of households in the US who didn't buy a car
logS0 = math.log(S0) # taking the log

# getting the market share, the log of it, and then creating our y variable
cars['marketShare'] = cars['Quantity']/nHH2005 
cars['log_marketShare'] = cars['marketShare'].apply(math.log)
cars['Y'] = cars['log_marketShare'] - logS0

carCat = pd.crosstab(index=cars['categoryName'], columns='count')
characteristics = ['mpg_combined','footprint', 'hp', 'disp', 'weight']
cars['categoryCount'] = cars['categoryName'].map(lambda x: carCat.loc[x,'count'])
featuresAvg = cars.groupby(['categoryName'])[characteristics].mean()

def dist2Cat(characteristics):
    #characteristics is a list of strings. Each string in the list is a name of a characteristic
    for ch in characteristics:
        # 1. expand                                          
        cars[ch+'Avg'] = cars['categoryName'].map(lambda x: featuresAvg[ch][x]) # can use .loc here
        
        # could do the next part in 1 line and not using lambda fn
        
        # 2. difference
        cars[ch+'Dist'] = cars[ch]-cars[ch+'Avg']
        # 3. square
        cars[ch+'Dist'] = cars[ch+'Dist'].map(lambda x: x*x)

dist2Cat(characteristics)

def dist2CatV2(characteristics):
    #characteristics is a list of strings. Each string in the list is a name of a characteristic
    for ch in characteristics:
        # 1. expand
        #cars[ch+'Avg'] = cars['categoryName'].map(lambda x: featuresAvg[ch][x])
        cars[ch+'Avg2'] = (cars[ch+'Avg']*cars['categoryCount'] - cars[ch])/(cars['categoryCount']-1)
        # 2. difference
        cars[ch+'Dist2'] = cars[ch]-cars[ch+'Avg2']
        # 3. square
        cars[ch+'Dist2'] = cars[ch+'Dist2'].map(lambda x: x*x)
        
dist2CatV2(characteristics)
cars.columns

features = ['Price', 'disp','hp', 'wheel_base', 'weight', 'mpg_combined', 'footprint', 'hpDist', 'dispDist', 'mpg_combinedDist', 'footprintDist']

fig = plt.figure(figsize = (9,9))
sns.heatmap(cars[features].corr(), annot = True, annot_kws={'size':10},vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()

model = smf.ols(formula = "Price ~ hp + mpg_combined + footprint + weight + C(category) + mpg_combinedDist + footprintDist",data=cars).fit(cov_type='HC1') # this end part makes it robust
print(model.summary())

model = smf.ols(formula = "Price ~ hp + mpg_combined + weight + mpg_combinedDist2 + C(category) + weightDist2",data=cars).fit(cov_type='HC1') # this end part makes it robust
print(model.summary())

cars['Price_hat'] = model.predict()

secondStageV1 = smf.ols(formula='Y ~ hp + mpg_combined + weight + Price_hat + C(category)',data=cars).fit(cov_type='HC1')
print(secondStageV1.summary())

import warnings
warnings.filterwarnings("ignore")
y_pred = secondStageV1.predict()

plt.figure(figsize = (7,5))
sns.distplot(cars['Y'], color = 'r', hist = False, label='Actual Values', kde_kws={"shade": False})
sns.distplot(y_pred, color = 'purple', hist = False, label = 'Predicted Values', kde_kws ={"shade": False} )
warnings.filterwarnings("ignore")
plt.title('Regression Distribution of Actual vs Predicted Values')
plt.legend()
plt.show()

corrs = cars.select_dtypes(include=np.number).astype('float').corr()
corrs['Y'].abs().sort_values(ascending = False)



