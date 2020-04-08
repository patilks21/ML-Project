
import pandas as pd
pd.options.display.max_columns = 999
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv('movie_metadata.csv')
data.dropna(how = 'any',axis = 0,inplace = True)

#Dividing into numerical and categorical data
numerical_features = data.select_dtypes(exclude=['object']).columns
categorical_features = data.select_dtypes(include=['object']).columns
num_data = data[numerical_features]
cat_data = data[categorical_features]

# Drop unnecessary columns
num_data.drop(['title_year'],axis = 1, inplace = True)
numerical_features = numerical_features.drop('title_year')

# Removing Outliers
import numpy as np
def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))

test = outliers_iqr(num_data['imdb_score'])
test = list(test)
num_data.drop(num_data.index[test],inplace = True)
cat_data.drop(cat_data.index[test],inplace = True)

#removing any instances that has less than 10k of number of voted users because number of voted users is most related to IMDB Score
a = num_data[(num_data.num_voted_users < 10000)].index
num_data.drop(a,inplace = True)
cat_data.drop(a,inplace = True)

# removing actor_3_facebook_likes, actor_1_facebook_likes, cast_total_facebook_likes, facenumber_in_poster, budget, actor_2_facebook_likes, aspect_ratio
num_data.drop([
 'actor_3_facebook_likes',
 'actor_1_facebook_likes',
 'cast_total_facebook_likes',
 'facenumber_in_poster',
 'budget',
 'actor_2_facebook_likes',
 'aspect_ratio'],inplace = True,axis = 1)

# Imputing NaN with Median (Because data has outliers)
num_data.fillna(num_data.median(),inplace = True)


# Categorical Variable Genre
df_genres = pd.DataFrame(cat_data['genres'])
df_genres = pd.DataFrame(df_genres.genres.str.split('|').tolist(),columns = ["Genre_"+str(i) for i in  range(0,8)] )
df_genres = df_genres.reindex(cat_data.index)

# droping the genres
cat_data.drop('genres',inplace = True, axis = 1)
cat_data = cat_data.merge(df_genres,left_index = True,right_index = True)

# droping plot keywords
df_plot_keywords = pd.DataFrame(cat_data['plot_keywords'])
df_plot_keywords = pd.DataFrame(df_plot_keywords.plot_keywords.str.split('|').tolist(),columns = ["plot_keywords_"+str(i) for i in  range(0,5)] )
cat_data.drop('plot_keywords',inplace = True, axis = 1)
df_plot_keywords = df_plot_keywords.reindex(cat_data.index)
cat_data = cat_data.merge(df_plot_keywords,left_index = True,right_index = True)

# Filling NaN values in Categorical Features with Mode
cat_data.drop(['movie_imdb_link','Genre_6','Genre_7'],inplace = True, axis = 1)

# Split Data before doing any transformations
whole_data = pd.concat([num_data,cat_data],axis = 1)
y = whole_data['imdb_score']
whole_data.drop('imdb_score',axis = 1,inplace = True)
from sklearn.model_selection import train_test_split # to split the data into two parts
X_train,X_test,y_train,y_test = train_test_split(whole_data,y, random_state = 0,test_size = 0.20) # test_size = 0.10
num_feat = whole_data.select_dtypes(exclude=['object']).columns.tolist()
cat_feat = whole_data.select_dtypes(include=['object']).columns.tolist()
X_train_num = X_train[num_feat]
X_train_cat = X_train[cat_feat]
X_test_num = X_test[num_feat]
X_test_cat = X_test[cat_feat]

# Handling Skewness
from scipy.stats import skew
skewness = X_train_num.apply(lambda x: skew(x.dropna()))
skewness = skewness[abs(skewness) > 0.75]
skew_features = X_train_num[skewness.index]
skew_features  = np.log1p(skew_features)
X_train_num[skewness.index] = skew_features

# Standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
for i, col in enumerate(num_feat):
    X_train_num.loc[:,col] = X_train_num_scaled[:, i]

# Transformations on Numeric Test data
from scipy.stats import skew
skewness = X_test_num.apply(lambda x: skew(x.dropna()))
skewness = skewness[abs(skewness) > 0.75]
skew_features = X_test_num[skewness.index]
skew_features  = np.log1p(skew_features)
X_test_num[skewness.index] = skew_features
X_test_num_scaled = scaler.transform(X_test_num)
for i, col in enumerate(num_feat):
    X_test_num.loc[:,col] = X_test_num_scaled[:, i]

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
dt = RandomForestRegressor(n_estimators = 1000,n_jobs=-1,random_state = 0)
dt.fit(X_train_num, y_train)
dt_score_train = dt.score(X_train_num, y_train)
print("Random Forest Regression :")
print("Accuracy on Training score: ",dt_score_train)
dt_score_test = dt.score(X_test_num, y_test)
print("Accuracy on Testing score: ",dt_score_test)

# One Hot Encoding of Categorical values
X_train_cat.drop(['Genre_2','Genre_3','Genre_4','Genre_5'],axis = 1,inplace = True)
X_test_cat.drop(['Genre_2','Genre_3','Genre_4','Genre_5'],axis = 1,inplace = True)

# Coupling last 5% values as "Other" in each Categorical Variable
temp_cat = pd.concat([X_train_cat,X_test_cat])
temp_cat.loc[temp_cat[~temp_cat["country"].isin(['USA',
 'UK',
 'France',
 'Germany'])].index,"country"] = "Other"
temp_cat["language"] = (temp_cat["language"] == "English") * 1
temp_cat.loc[temp_cat[(temp_cat["content_rating"] != "R")&(temp_cat["content_rating"] != "PG-13")&(temp_cat["content_rating"] != "PG")].index,"content_rating"] = "Other"
temp_cat.loc[temp_cat[(temp_cat["Genre_0"] != "Action")&(temp_cat["Genre_0"] != "Drama")&(temp_cat["Genre_0"] != "Comedy")&(temp_cat["Genre_0"] != "Adventure")&(temp_cat["Genre_0"] != "Crime")&(temp_cat["Genre_0"] != "Biography")].index,"Genre_0"] = "Other"
temp_cat.Genre_1.value_counts()
temp_cat.Genre_1.value_counts().index.tolist()
temp_cat.loc[temp_cat[~temp_cat["Genre_1"].isin(['Drama',
 'Adventure',
 'Crime',
 'Comedy',
 'Romance',
 'Mystery',
 'Thriller',
 'Horror',
 'Family',
 'Animation',
 'Fantasy'])].index,"Genre_1"] = "Other"
temp_cat["color"] = (temp_cat["color"] == "Color") * 1
temp_cat.drop(['movie_title'],inplace = True, axis = 1)

# LabelEncoder for rest of the Categorical Values (Have levels > 1000)
temp_cat = pd.get_dummies(temp_cat)
X_train_cat = temp_cat.loc[X_train_cat.index,:]
X_test_cat = temp_cat.loc[X_test_cat.index,:]
X_train = pd.concat([X_train_num,X_train_cat], axis =1)
X_test = pd.concat([X_test_num,X_test_cat], axis =1)

# Classifying a movie into 4 classes based on IMDB Score
temp_whole = pd.concat([X_train,X_test])
target = pd.concat([y_train,y_test])
target_classes = pd.cut(target,bins = [0,6,10],labels = [0,1],right = True,include_lowest = True)
target_classes.value_counts()
from sklearn.model_selection import train_test_split # to split the data into two parts
X_train,X_test,y_train,y_test = train_test_split(temp_whole,target_classes, random_state = 1,test_size = 0.20,stratify =target_classes) # test_size = 0.10

# Logistic Regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train,y_train)
print('\nLogistic regression classifier')
print('Accuracy on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
dt = RandomForestClassifier(n_estimators = 1000,n_jobs=-1,random_state = 0)
dt.fit(X_train, y_train)
dt_score_train = dt.score(X_train, y_train)
print('\nRandom Forest Classifier')
print("Accuracy on Training score: ",dt_score_train)
dt_score_test = dt.score(X_test, y_test)
print("Accuracy on Testing score: ",dt_score_test)

