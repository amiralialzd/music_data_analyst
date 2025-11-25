
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
url="/Users/amirali/PycharmProjects/PythonProject/music_recommendation_system/archive/Music Recommendation System using Spotify New Dataset.csv"
df=pd.read_csv(url)
#the type_of_tempo_in_Beats_Per_Minute(BPM) has been dropped because it had lots of missing values
df.drop(axis=1,columns="type_of_tempo_in_Beats_Per_Minute(BPM)",inplace=True)

# replacing missing values with the most frequent value in the type_of_song_by_loudness(dB) column because it is an object type column
mostFrequence=df["type_of_song_by_loudness(dB)"].value_counts().idxmax()
print(mostFrequence)
df["type_of_song_by_loudness(dB)"]=df["type_of_song_by_loudness(dB)"].replace(np.nan,mostFrequence)

df.drop(axis=1,columns=["year","artists","id"],inplace=True)

print(df.isnull().sum())
df_num=df[["danceability", "energy", "acousticness", "tempo", "duration_min", "liveness", "speechiness", "instrumentalness","popularity"]]
corr=df_num.corr()
print(corr)
#taking one hot from all categorical columns
one_hot_type_of_song_by_danceability=pd.get_dummies(df["type_of_song_by_danceability"],dtype=int,drop_first=True)
one_hot_type_of_song_by_valence=pd.get_dummies(df["type_of_song_by_valence"],dtype=int,drop_first=True)
one_hot_type_of_song_by_acousticness=pd.get_dummies(df["type_of_song_by_acousticness"],dtype=int,drop_first=True)
one_hot_type_of_Energy_Level_of_the_Song=pd.get_dummies(df["Energy_Level_of_the_Song"],dtype=int,drop_first=True)
one_hot_type_of_type_of_song_by_instrumentalness=pd.get_dummies(df["type_of_song_by_instrumentalness"],dtype=int,drop_first=True)
one_hot_type_of_type_of_song_by_liveness=pd.get_dummies(df["type_of_song_by_liveness"],dtype=int,drop_first=True)
one_hot_type_of_type_of_type_of_song_by_loudness=pd.get_dummies(df["type_of_song_by_loudness(dB)"],dtype=int,drop_first=True)
one_hot_type_of_type_of_type_of_type_of_song_by_mode=pd.get_dummies(df["type_of_song_by_mode"],dtype=int,drop_first=True)
one_hot_type_of_type_of_type_of_song_by_speechiness=pd.get_dummies(df["type_of_song_by_speechiness"],dtype=int,drop_first=True)
df_cat=df["popularity"]
df_cat=pd.concat([df_cat,one_hot_type_of_type_of_type_of_song_by_speechiness,one_hot_type_of_type_of_type_of_type_of_song_by_mode,one_hot_type_of_type_of_type_of_song_by_loudness,one_hot_type_of_type_of_song_by_liveness,one_hot_type_of_type_of_song_by_instrumentalness,one_hot_type_of_Energy_Level_of_the_Song,one_hot_type_of_song_by_acousticness,one_hot_type_of_song_by_valence,one_hot_type_of_song_by_danceability],axis=1)
#print(df_cat.head())
#print(one_hot_type_of_Energy_Level_of_the_Song)
#corr_cat=df_cat.corr()
#print(corr_cat)
#slow song from energy level of the song and mostly acoustic song have strong negative correlation with popularity so we will add them to our df_clean plus our numerical columns
df_clean=df[["popularity","acousticness","energy"]]
df_clean=pd.concat([df_clean,df_cat.drop(columns="popularity",axis=1),df[["danceability","tempo","duration_min","speechiness","instrumentalness"]]],axis=1)
print(df_clean.info())
#showing box plot for popularity with slow song which came from get dummies function for energy level
sns.boxplot(data=df_clean,y="popularity",x="Slow song")
#plt.show()
#plt.close()

#showing scatterplot with acousticness (numerical column)
sns.scatterplot(data=df_clean,y="popularity",x="acousticness",color="red")
#plt.show()
#plt.close()
plt.hist(df_clean["popularity"],color="skyblue",edgecolor="black")
plt.title("popularity values and frequency")
plt.xlabel("popularity values")
plt.ylabel("frequency")
#plt.show()
#plt.close()

#train test split part
x_train,x_test,y_train,y_test=train_test_split(df_clean.drop(columns="popularity",axis=1),df_clean["popularity"],test_size=0.3,random_state=1)
lr=LinearRegression()
lr.fit(x_train,y_train)
yhtat=lr.predict(x_test)
r2_train=lr.score(x_train,y_train)
r2_test=lr.score(x_test,y_test)
print(f"r2 train {r2_train} and r2 test is {r2_test}")
mse=mean_squared_error(y_test,yhtat)
print(f"mean square error is {mse}")
coe=lr.coef_
print(f"the coefficient is {coe.mean()}")

#pipline
Input=[("Scale",StandardScaler()),("Polynomial",PolynomialFeatures()),("model",LinearRegression())]
pip=Pipeline(Input)
pip.fit(x_train,y_train)
pipPredict=pip.predict(x_test)

#Ridge
ridge=Ridge(alpha=2)
ridge.fit(x_train,y_train)
ridge.predict(x_test)

#Gridsearch
parameter=[{"alpha":[1,2,3,4,5,6,7,8,9,10]}]
gRidge=Ridge()
grid=GridSearchCV(gRidge,parameter,cv=4)
gridfit=grid.fit(x_train,y_train)
bestS=gridfit.best_estimator_
print(f"best stimator is {bestS}")

#cross validation
validation=cross_val_score(bestS,x_train,y_train,cv=4)
print(f"cross validation is {validation.mean()}")



# Ridge model with the best stimator
bestRidge=Ridge(alpha=3)
bestRidge.fit(x_train,y_train)
coe=bestRidge.coef_
ridge_pred = bestRidge.predict(x_test)
print(f"the coefficient is {coe.mean()}")
plt.figure(figsize=(8,6))
plt.scatter(y_test, ridge_pred, alpha=0.4)

# Perfect prediction line
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linewidth=2)
plt.title("Ridge Regression: Actual vs Predicted Popularity")
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.show()






