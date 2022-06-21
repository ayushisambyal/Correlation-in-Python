# CORRELATIONS IN PYTHON !

# In this project, we read a csv file having all the data of a list of movies. 
# Then we perform Data wrangling, look for missing data & find correlations in the Data.

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot') # gives style to the graph
from matplotlib.pyplot import figure

# helps in rendering the figure
matplotlib.rcParams['figure.figsize'] = (10,6)

pd.options.mode.chained_assignment = None

df = pd.read_csv(r'D:\Project\7. Python Movie project\movies.csv')
df 

# Looking for missing data using loops
for col in df.columns:
    percent_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(percent_missing*100, 2)))

# printing Data Types for our columns
print(df.dtypes)

# Create correct year column
df['yearcorrect'] = df['released'].astype(str).str[:4]

# sorting by gross
df.sort_values(by=['gross'], inplace=False, ascending=False)
# displaying max rows 
pd.set_option('display.max_rows', None)

# Drop any duplicates
df.drop_duplicates()

# Budget high correlation
# Company high correlation

# Scatter plot with budget vs gross
plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget for Film')
plt.show()

# Plot budget vs graph using seaborn
sns.regplot(x='budget', y='gross', data=df, scatter_kws={"color":"green"}, line_kws={"color":"blue"})
plt.show()

# Finding correlation 
df.corr(method ='pearson')  # pearson, kendall, spearman

# High correlation between budget & gross
correlation_matrix = df.corr(method = 'pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()

#looks at Company 
df.head()

df_numerized = df
for col in df_numerized.columns:
    if(df_numerized[col].dtype == 'object'):
        df_numerized[col] = df_numerized[col].astype('category')
        df_numerized[col] = df_numerized[col].cat.codes # gives random numerization
df_numerized.head()

correlation_matrix = df_numerized.corr(method = 'pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()

df_numerized.corr()

# unstacking
correlation_mat = df_numerized.corr()
corr_pairs = correlation_mat.unstack()
corr_pairs

sorted_pairs = corr_pairs.sort_values()
sorted_pairs

high_corr = sorted_pairs[(sorted_pairs)>0.5]
high_corr

# Votes & budget have the highest correlation to gross earnings
# Company has low correlation
