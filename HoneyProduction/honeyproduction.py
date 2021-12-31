import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Reads in the data for honey production in the US between 1998 and 2012.
df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

# Prints the first 5 lines of the DataFrame to give an idea of the data structure.
# print(df.head())
#	state	numcol	yieldpercol	totalprod	   stocks	    priceperlb	prodvalue	   year
#0	AL	16000.0	  71	      1136000.0	   159000.0	  0.72	      818000.0	   1998
#1	AZ	55000.0	  60	      3300000.0	   1485000.0	0.64	      2112000.0	   1998
#2	AR	53000.0	  65	      3445000.0	   1688000.0	0.59	      2033000.0	   1998
#3	CA	450000.0	83	      37350000.0	 12326000.0	0.62	      23157000.0	 1998
#4	CO	27000.0	  72	      1944000.0	   1594000.0	0.7	        1361000.0	   1998

# Gets a smaller DataFrame of the average of total honey production per year.
prod_per_year = df.groupby("year")["year", "totalprod"].mean()
# print("Annual production per year is:", prod_per_year)
#year	totalprod
#1998	5105093.023255814
#1999	4706674.4186046515
#2000	5106000.0
#2001	4221545.454545454
#2002	3892386.3636363638
#2003	4122090.909090909
#2004	4456804.87804878
#2005	4243146.341463415
#2006	3761902.43902439
#2007	3600512.1951219514
#2008	3974926.8292682925
#2009	3626700.0
#2010	4382350.0
#2011	3680025.0
#2012	3522675.0

# Gets the year column, w/o numbering rows.
X = prod_per_year['year']
X = X.values.reshape(-1, 1)

# Gets the average state total production column.
y = prod_per_year['totalprod']
y = y.values.reshape(-1, 1)

# Creates a linear regression model on the data.
regr = linear_model.LinearRegression()
regr.fit(X, y)
print("Honey production is decreasing at ", regr.coef_[0][0], "lbs/year.")
y_predict = regr.predict(X)

# Prepares the scatterplot of the values.
plt.scatter(X, y, color="gold")
plt.plot(X, y_predict, "-", color="salmon")
plt.xlabel('Year')
plt.ylabel('Avg. State Honey Production (US)')
plt.title('Avg. State Honey Production (lbs) vs Year')

# Creates an array of years from 2013 to 2050
X_future = np.array(range(2013, 2050))
# Transforms from a row of values to a column for scikit-learn.
X_future = X_future.reshape(-1, 1)

# Predicts the 2013 - 2050 values using our model and plots it. 
future_predict = regr.predict(X_future)
plt.plot(X_future, future_predict)
plt.show()

print("Honey production predicted in 2050: ", future_predict[(2050-2013-1)][0], "lbs.")




