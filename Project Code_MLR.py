####################################
# Import all the necessary libraries
####################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as smf 
import xgboost as xgb
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor


######################################
# Load the given train & test data set
######################################
inp_data = pd.read_csv("C:/Users/Admin/Desktop/Data Science/Project Search/Datasets/Project_Data.csv")
sel_data = inp_data.iloc[:,2:]
sel_data.dtypes

# Transform Categorical columns (object type) to the numerical data type using Label Encoder
cols = ['Course Category','Course Type','Course Track','Course Level','Course Curriculam','Course Language','Course Parttner/Institution','Assessment Method','Certificate Type','Location','Seasoned Faculty','Internship','Career Guidance']
sel_data[cols] = sel_data[cols].apply(LabelEncoder().fit_transform)

sel_data.dtypes
sel_data.describe(include='all')

##############################################################
# Perform Exploratory Data Analysis (EDA) for the inut dataset
##############################################################
#1. Check any null values in the dataset
########################################
sel_data.isnull().sum()
#Inference: No null values in the dataset

#####################################
#2. identify the outlier thru boxplot
#####################################
#selected the necessary columns which has multiple range of values
column1 = ['Course Type','Course Duration','Course Level','Instructor Rating','Course Curriculam']
sel_data[column1].boxplot()
#Inference: Few Outliers available in Rating & Course duration which is not able to eliminate (duration is more for Classroom session, Rating will be 0 for few)

####################################################################################
#3a. Check the Mean, Median value for measuring the center and distribution of values
####################################################################################
print ("Mean Values in the Distribution")
print (sel_data.mean())
print ("*******************************")
print ("Median Values in the Distribution")
print (sel_data.median())
#Inference: There is no much deviation of Mean & Median values for the input variables. 

sel_data.var()

##########################################
#3b. Check the Third & Fourth moment values
##########################################
# Third moment business decision
sel_data.skew()

#Fourth moment business decision
sel_data.kurt()


############################################################
#4. Check the count thru bar chart for the necessary columns
############################################################ 
plt.subplot(221)
sel_data['Course Type'].value_counts().plot(kind='bar', title='Course Type', figsize=(16,9))
plt.xticks(rotation=0)

plt.subplot(222)
sel_data['Course Duration'].value_counts().plot(kind='bar', title='Course Duration', figsize=(16,9))
plt.xticks(rotation=0)

plt.subplot(223)
sel_data['Course Level'].value_counts().plot(kind='bar', title='Course Level', figsize=(16,9))
plt.xticks(rotation=0)

plt.subplot(224)
sel_data['Course Curriculam'].value_counts().plot(kind='bar', title='Course Curriculam', figsize=(16,9))
plt.xticks(rotation=0)

##############################################################
#5. Analyzing Categorical variable using User-defined function
##############################################################
# Function analyzing categorical variables examined in each of category are showing counts of category and percentage of presence.
def cat_variables(dataframe, columns_name, plot=False):
    if dataframe[columns_name].dtypes == "bool":
        dataframe[columns_name] = dataframe[columns_name].astype(int)
        print(pd.DataFrame({columns_name: dataframe[columns_name].value_counts(),
                          "Ratio": 100 * inp_data[columns_name].value_counts() / len(dataframe)})) 
        print("############################################################################")
        if plot:
            sns.countplot(x=dataframe[columns_name], data=dataframe)
            plt.show(block=True)
    else:
        print(pd.DataFrame({columns_name: dataframe[columns_name].value_counts(),
                          "Ratio": 100 * inp_data[columns_name].value_counts() / len(dataframe)})) 
        if plot:
          sns.countplot(x=dataframe[columns_name], data=dataframe)
          plt.show(block=True)

            
cat_variables(inp_data,"Course Type", plot=True)
cat_variables(inp_data,"Course Track", plot=True)
cat_variables(inp_data,"Course Level", plot=True)
cat_variables(inp_data,"Course Curriculam", plot=True)
cat_variables(inp_data,"Course Language", plot=True)
cat_variables(inp_data,"Assessment Method", plot=True)
#Inference: Examined categorical variables significance with the counts and percentage.

###############################################
#6a. Analyzing Numerical variable thru histogram
###############################################
def num_variables(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
        
num_variables(inp_data,"Course Duration", plot=True)
num_variables(inp_data,"Instructor Rating", plot=True)
num_variables(inp_data,"Price in INR", plot=True)
#Inference: Duration is right skewed with avg hrs of 7, Rating is left skewed with avg of 4.2 and Price looks like bi-modal shaped curve (i.e 2 peaks)

##############################################
#6b. Bivariant analysis for Numerical variable 
##############################################
sns.scatterplot(data=sel_data, x='Course Duration',y='Price in INR')
sns.scatterplot(data=sel_data, x='Instructor Rating',y='Price in INR')

#######################################
#6c. Identify the Outliers thru Boxplot
#######################################
fig, ax = plt.subplots(2, 2, figsize = (15, 13))
sns.boxplot(x= sel_data["Course Duration"], ax = ax[0,0])
sns.distplot(sel_data['Course Duration'], ax = ax[0,1])
sns.boxplot(x= sel_data["Instructor Rating"], ax = ax[1,0])
sns.distplot(sel_data['Instructor Rating'], ax = ax[1,1])

######################################################################
#7a. Check the correlated values and remove the high correlated columns
######################################################################
def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={'figure.figsize': (16, 16)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(sel_data, plot=True)
#Inference: Career Guidance & Internship column has high correlation and hence, removed Career Guidance & kept only Internship

############################################################################
#7b. Validate the multi-linearity among the feature variables using the VIFs
############################################################################
new = sel_data.drop(['Career Guidance','Instructor Rating','Assessment Method','Course Category'],1)
vif = pd.DataFrame()
vif['Features'] = new.columns
vif['VIF'] = [variance_inflation_factor(new.values, i) for i in range(new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
#Inference: Career Guidance, Instructor Rating & Assessment Method has high VIF values

############################################################################
#7c. Validate the multi-linearity among the feature variables using pairplot
############################################################################
sns.set(rc={'figure.figsize':(11.7,8.27)})
Data_attr = sel_data.iloc[:, :]
sns.pairplot(Data_attr, diag_kind='kde') 
#Inference: Course Duration, Instructor Rating & Institution has high dependency

########################################
#7d. Validate the p-value with OLS model
########################################
y_value = sel_data['Price in INR']
X_value = sel_data.drop('Price in INR',1)

ml1 = smf.OLS(y_value, X_value).fit() # regression model
ml1.summary()

#Inference: Course Track, Course Curriculam & Course Level has high p-value
# Based on VIF & Model summary, few colmns are dropped which are still insignificant to the model
sel_data.drop(['Career Guidance','Course Track','Course Curriculam','Course Level'],1,inplace=True)

######################################################
# Build and Evaluate the Multi-Linear Regression model
######################################################
X_value = sel_data
final_model = smf.OLS(y_value, X_value).fit()
final_model.summary()

#########################################################
# Model Evaluation
# Splitting the input randomly into train and test data
######################################################### 
train, test = train_test_split(sel_data, test_size = 0.20, random_state=25) # 20% test data

y_train = train['Price in INR']
X_train = train.drop('Price in INR',1)
y_test = test['Price in INR']
X_test = test.drop('Price in INR',1)

# Preparing the model on train data 
trained_model = smf.OLS(y_train, X_train).fit()

#############################
# Prediction on test data set
############################# 
test_pred = trained_model.predict(X_test)
test_resid = test_pred - y_test

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse
test_r2 = r2_score(y_true = y_test, y_pred = test_pred)

#Inference: Arrived the Test prediction R-Squared value of 0.817 and RMSE value of +/- 1292

##############################
# Prediction on train data set
############################## 
train_pred = trained_model.predict(X_train)
train_resid = train_pred - y_train

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
train_r2 = r2_score(y_true = y_train, y_pred = train_pred)

print("\nLinear Regression Model............................................\n")
print ("Test Accuracy with OLS: ", test_r2)
print ("Test RMSE with OLS: ", test_rmse)
print ("Train Accuracy with OLS ", train_r2)
print ("Train RMSE with OLS: ", train_rmse)

#Inference: Arrived the Train prediction R-Squared value of 0.849 and RMSE value of +/- 1198

################## Model Performance ##################################
# Both the Test & Train predicted values are approximately nearby value
# hence, the model is best fitted with the given dataset
# Model accuracy is 81.7%
#######################################################################

################################################
######### Model 2 - Ridge Regression ###########
################################################
# Identify best fit/parameter for Ridge Regression
ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 5, 10, 20]}
ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(sel_data.iloc[:, 0:11], sel_data["Price in INR"])
ridge_param = ridge_reg.best_params_
ridge_reg.best_score_
ridge_pred = ridge_reg.predict(sel_data.iloc[:, 0:11])
ridge_reg.score(sel_data.iloc[:, 0:11], sel_data["Price in INR"])
np.sqrt(np.mean((ridge_pred - sel_data["Price in INR"])**2))

# Create the Ridge model with identified best parameter 
ridge = Ridge(alpha= ridge_param['alpha'])
ridge.fit(X_train,y_train)

# Predict the score for test dataset
ridge_test_pred = ridge.predict(X_test)
test_ridge = ridge.score(X_test,y_test)
test_rid_rmse = np.sqrt(np.mean((ridge_test_pred - y_test)**2))

# Predict the score for train dataset
ridge_train_pred = ridge.predict(X_train)
train_ridge = ridge.score(X_train,y_train)
train_rid_rmse = np.sqrt(np.mean((ridge_train_pred - y_train)**2))

print("\nRidge Model............................................\n")
print("Test Accuracy with Ridge: ",test_ridge)
print("Test RMSE with Ridge: ",test_rid_rmse)
print("Train Accuracy with Ridge: ",train_ridge)
print("Train RMSE with Ridge: ",train_rid_rmse)

#Inference: 
# Arrived the Test prediction R-Squared value of 0.819 and RMSE value of +/- 1287
# Arrived the Train prediction R-Squared value of 0.851 and RMSE value of +/- 1189

################################################      
######### Model 3 - Lasso Regression ###########
################################################
# Identify best fit/parameter for Lasso Regression
lasso = Lasso()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 2, 5, 10, 20]}
lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(sel_data.iloc[:, 0:11], sel_data["Price in INR"])
lasso_param = lasso_reg.best_params_
lasso_reg.best_score_
lasso_pred = lasso_reg.predict(sel_data.iloc[:, 0:11])
lasso_reg.score(sel_data.iloc[:, 0:11], sel_data["Price in INR"])
np.sqrt(np.mean((lasso_pred - sel_data["Price in INR"])**2))

# Create the Lasso model with identified best parameter 
lasso = Lasso(lasso_param['alpha'], normalize = True)
lasso.fit(X_train,y_train)
lasso.coef_
lasso.intercept_

# Predict the score for test dataset
lasso_test_pred = lasso.predict(X_test)
test_las_score = lasso.score(X_test,y_test)
test_las_rmse = np.sqrt(np.mean((lasso_test_pred - y_test)**2))

# Predict the score for train dataset
lasso_train_pred = lasso.predict(X_train)
train_las_score = lasso.score(X_train,y_train)
train_las_rmse = np.sqrt(np.mean((lasso_train_pred - y_train)**2))

print("\nLasso Model............................................\n")
print ("Test Accuracy with Lasso: ", test_las_score)
print ("Test RMSE with Lasso: ", test_las_rmse)
print ("Train Accuracy with Lasso: ", train_las_score)
print ("Train RMSE with Lasso: ", train_las_rmse)

#Inference: 
# Arrived the Test prediction R-Squared value of 0.814 and RMSE value of +/- 1304
# Arrived the Train prediction R-Squared value of 0.843 and RMSE value of +/- 1221

#####################################################
######### Model 4 - ElasticNet Regression ###########
#####################################################
# Identify the best fit/parameter for ElasticNet Regression
from sklearn.model_selection import RepeatedKFold
enet = ElasticNet()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1, 2, 5, 10, 20],'l1_ratio': np.arange(0, 1, 0.01)}
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_absolute_error', cv = cv)
enet_reg.fit(sel_data.iloc[:, 0:11], sel_data["Price in INR"])
enet_param = enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(sel_data.iloc[:, 0:11])
enet_reg.score(sel_data.iloc[:, 0:11], sel_data["Price in INR"])
np.sqrt(np.mean((enet_pred - sel_data["Price in INR"])**2))

# Create the ElasticNet model with identified best parameter 
enet_model = ElasticNet(enet_param['alpha'], normalize = True)
#enet = ElasticNet(alpha=0.00001, normalize = True)
enet_model.fit(X_train,y_train)
enet_model.coef_
enet_model.intercept_

# Predict the score for test dataset
enet_test_pred = enet_model.predict(X_test)
test_enet_score = enet_model.score(X_test,y_test)
test_enet_rmse = np.sqrt(np.mean((enet_test_pred - y_test)**2))

# Predict the score for train dataset
enet_train_pred = enet_model.predict(X_train)
train_enet_score = enet_model.score(X_train,y_train)
train_enet_rmse = np.sqrt(np.mean((enet_train_pred - y_train)**2))

print("\nElasticNet Model............................................\n")
print ("Test Accuracy with ElasticNet: ", test_enet_score)
print ("Test RMSE with ElasticNet: ", test_enet_rmse)
print ("Train Accuracy with ElasticNet: ", train_enet_score)
print ("Train RMSE with ElasticNet: ", train_enet_rmse)

#####################################################
######### Boosting the Models for accuracy ##########
#####################################################
#create Gradient Booster model
gb_reg = GradientBoostingRegressor(
     n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0).fit(X_train, y_train)
mean_squared_error(y_test, gb_reg.predict(X_test))
test_gb_score = gb_reg.score(X_test,y_test)
test_gb_rmse = np.sqrt(np.mean((gb_reg.predict(X_test) - y_test)**2))
train_gb_score = gb_reg.score(X_train,y_train)
train_gb_rmse = np.sqrt(np.mean((gb_reg.predict(X_train) - y_train)**2))

print("\nGradient Boosting Model............................................\n")
print ("Test Accuracy with Gradient Boosting: ", test_gb_score)
print ("Test RMSE with Gradient Boosting: ", test_gb_rmse)
print ("Train Accuracy with Gradient Boosting: ", train_gb_score)
print ("Train RMSE with Gradient Boosting: ", train_gb_rmse)

#create XGradient Booster model
xgb_reg = xgb.XGBRegressor(max_depths = 100, n_estimators = 100, learning_rate = 0.1, n_jobs = -1)
xgb_reg.fit(X_train, y_train)
test_xgb_score = xgb_reg.score(X_test,y_test)
test_xgb_rmse = np.sqrt(np.mean((xgb_reg.predict(X_test) - y_test)**2))
train_xgb_score = xgb_reg.score(X_train,y_train)
train_xgb_rmse = np.sqrt(np.mean((xgb_reg.predict(X_train) - y_train)**2))

print("\nXGradient Boosting Model............................................\n")
print ("Test Accuracy with XGradient Boosting: ", test_xgb_score)
print ("Test RMSE with XGradient Boosting: ", test_xgb_rmse)
print ("Train Accuracy with XGradient Boosting: ", train_xgb_score)
print ("Train RMSE with XGradient Boosting: ", train_xgb_rmse)

#combine all models and ensemble
estimators = [('ridge', Ridge(ridge_param['alpha'])),
              ('lasso', Lasso(lasso_param['alpha'],random_state=42)),
              ('enet', ElasticNet(enet_param['alpha'], normalize = True)),
              ('linear', LinearRegression()),
              ('knr', KNeighborsRegressor(n_neighbors=20,
                                         metric='euclidean'))]

final_estimator = GradientBoostingRegressor(
    n_estimators=100, subsample=0.5, min_samples_leaf=25, max_features=1, learning_rate=0.1, max_depth=1,
    random_state=42)
from sklearn.ensemble import StackingRegressor
reg = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator)

reg.fit(X_train, y_train)
ens_pred = reg.predict(X_test)
print('Ensemble score: {:.2f}'.format(r2_score(y_test, ens_pred)))

final_estimator1 = xgb.XGBRegressor(
    max_depths = 100, n_estimators=100, subsample=0.5, min_samples_leaf=25, max_features=1, learning_rate=0.1, 
    random_state=42)
from sklearn.ensemble import StackingRegressor
reg = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator1)

reg.fit(X_train, y_train)
ens_pred = reg.predict(X_test)
print('Ensemble score: {:.2f}'.format(r2_score(y_test, ens_pred)))
