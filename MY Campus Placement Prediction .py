#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt


# In[10]:


df = pd.read_csv("D:\\4. Projects\\Campus Placement Prediction\\archive\\Placement_data_full_class.csv")
df.head()


# In[11]:


df.info()


# In[12]:


#seprating categorical and numerical values 


# In[13]:


catvar = list(df.select_dtypes(include = ['object']).columns)

numvar = list(df.select_dtypes(include = ['int32' , 'int64' ,'float32' , 'float64']).columns)


# In[14]:


catvar 


# In[15]:


numvar


# In[16]:


df.isnull().sum()


# In[17]:


# Handling the null values 


# In[18]:


df.shape


# In[19]:


def plotdistplot(col):
    
    plt.figure(figsize=(15,7))
    sn.distplot(df['salary'], kde=True , hist = False , label='actual salary', color ='orange')
    sn.distplot(df[col], kde=True , hist = False , label= col, color ='black')


# In[20]:


df['salary'].mode()[0]


# In[21]:


df['salary_mean'] = df['salary'].fillna(df['salary'].mean())
df['salary_mode'] = df['salary'].fillna(df['salary'].mode()[0])
df['salary_median'] = df['salary'].fillna(df['salary'].median())

df.head()


# In[22]:


#plotting


# In[23]:


sallist = ['salary_mean' ,'salary_mode' , 'salary_median']
for sal in sallist:
    plotdistplot(sal)


# In[24]:


#by understanding the graph we decided to fill tha null by mean values


# In[25]:


df['salary'] =df['salary'].fillna(df['salary'].mean())

df.head()


# In[26]:


df.isnull().sum()


# In[27]:


#eda Exploratory Data Analysis


# In[28]:


sn.countplot(df['status'], palette = 'plasma')


# In[29]:


def valuecount(col):
    return dict(df[col].value_counts())

def getcountplot(col):
    
    sn.countplot(df[col] , palette = 'plasma')
    plt.yticks(fontweight='bold' , color = 'blue')
    plt.xticks(fontweight='bold' , color = 'blue')
    plt.show()
    
for col in catvar:
    print(f'count plot for feature {col} is show down')
    getcountplot(col)
    print('=*75')


# In[30]:


getfinaldict = {}
for col in catvar:
    getfinaldict[col] = valuecount(col)
    
getfinaldict


# In[31]:


#top science student placed


# In[32]:


df[(df['degree_t']=='Sci&Tech')&(df['status']=='Placed')].sort_values(by = 'salary',ascending = False).reset_index().head(5)


# In[33]:


#top Comm&Mgmt student placed


# In[34]:


df[(df['degree_t']=='Comm&Mgmt')&(df['status']=='Placed')].sort_values(by = 'salary' , ascending = False).reset_index().head(5)


# In[35]:


#min and max salary of student 


# In[36]:


df[(df['salary']==max(df['salary'])) | (df['salary']==min(df['salary']))].sort_values(by='salary', ascending = False).reset_index().head(5)


# In[37]:


#salary more than average in science and technology 


# In[38]:


df[(df['degree_t']=='Sci&Tech')&(df['salary']>df['salary'].mean())].sort_values(by ='salary' ,ascending = False).reset_index().head(5)


# In[39]:


#max student placed are from 'Comm&Mgmt'


# In[40]:


df.groupby(['degree_t'])['status'].count().plot(kind = 'bar' , color = 'wheat')


# In[41]:


df.info()


# In[42]:


df.groupby(['gender'])['status'].count().plot(kind = 'pie' ,autopct='%1.1f%%')


# In[43]:


genf = df[df['gender']=='F']
genf[genf['salary']==max(genf['salary'])].style.background_gradient(cmap= 'plasma')


# In[44]:


df.groupby(['hsc_s'])['status'].count().plot(kind = 'pie' , autopct = '%1.1f%%', textprops={'weight': 'bold'})


# In[45]:


df


# In[46]:


def labelencode(le,col):
    
    df[col] = le.fit_transform(df[col])
    
    
getmappings = {}
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in list(getfinaldict.keys()):
    labelencode(le,col)
    getmappings[col] = dict(zip(le.classes_,le.transform(le.classes_)))
    

df.head()


# # Numerical Column EDA 

# In[47]:


numvar[1:]


# In[48]:


import scipy.stats as stat


# In[49]:


def getplots(df , col):
    
    plt.figure(figsize = (15,7))
    plt.subplot(1,3,1)
    plt.hist(df[col] , color = 'blueviolet' , edgecolor = 'black')
    
    plt.subplot(1,3,2)
    stat.probplot(df[col] , dist = 'norm' , plot =plt)
    
    plt.subplot(1,3,3)
    sn.boxplot(df[col], color = 'magenta')
    
for col in numvar[1:]:
    print(f'Distribution plots for col : {col} are shown ↓')
    getplots(df,col)
    print('='*10)


# In[50]:


data = df.copy()
test = data['status']
train = data.drop(['status','salary'],axis = 1)
train.head(5)
df = df.drop('salary_mean', axis=1)
df = df.drop('salary_mode', axis=1)
df = df.drop('salary_median', axis=1)


# In[51]:


## extratrees classifier

from sklearn.ensemble import ExtraTreesClassifier
ec = ExtraTreesClassifier()
ec.fit(train,test)


# In[52]:


featbar = pd.Series(ec.feature_importances_,index=train.columns)
featbar.nlargest(7).plot(kind = 'barh')


# In[53]:


featbar.nlargest(7).plot(kind='pie',autopct='%1.0f%%',figsize = (15,7))


# In[54]:


# mutual classif

from sklearn.feature_selection import mutual_info_classif
mc = mutual_info_classif(train,test)
ax = pd.Series(mc,index=train.columns)
ax.nlargest(7).plot(kind = 'barh')


# In[55]:


pd.Series(mc,index=train.columns).plot(kind = 'pie',autopct='%1.0f%%',figsize = (15,7))


# In[56]:


list(featbar.nlargest(10).index)


# In[57]:


list(ax.nlargest(10).index)


# In[58]:


featcol = list(featbar.nlargest(10).index)
mutclasif = list(ax.nlargest(10).index)
commoncols = list(set(featcol).intersection(set(mutclasif)))
print(commoncols)


# In[59]:


# selecting the common cols and will do training on these cols!
# these cols were selected as a nlargest result of 2 feature selection techniques!
df


# In[60]:


getmappings


# In[61]:


train = train[['gender','specialisation','degree_t','workex','ssc_p','hsc_p','degree_p','mba_p']]
train.columns = ['Gender','Specialisation','Techinal Degree','Work Experience','SSC_p','High School_p','Degree_p','MBA_p']
train.head()


# In[62]:


train['Techinal Degree'].value_counts()


# # MODEL BUILDING

# In[63]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,RandomizedSearchCV,cross_val_score
from sklearn import metrics
import pandas as pd
import numpy as np


# In[64]:


X_train, X_test, y_train, y_test = train_test_split(train,test,test_size=0.2)
X_train.shape,X_test.shape


# In[65]:


dc = DecisionTreeClassifier()
dc.fit(X_train,y_train)
plt.figure(figsize=(16,9))
tree.plot_tree(dc,filled=True,class_names=['Not_placed','Placed'],feature_names=train.columns)


# In[66]:


costpath = dc.cost_complexity_pruning_path(train,test)
ccp_alpha,impurities = costpath.ccp_alphas,costpath.impurities


# In[67]:


modellist = []
for alpha in ccp_alpha:
    dc = DecisionTreeClassifier(ccp_alpha=alpha)
    dc.fit(X_train,y_train)
    modellist.append(dc)
    


# In[69]:


train_score = [clf.score(X_train,y_train) for clf in modellist]
test_score = [clf.score(X_test,y_test) for clf in modellist]

plt.xlabel('alpha_value')
plt.ylabel('accuracy')
plt.plot(ccp_alpha,train_score,label = 'train_score',marker = '+',color = 'magenta')
plt.plot(ccp_alpha,test_score,label = 'test_score',marker = '*',color = 'red')
plt.legend()
plt.show()


# In[70]:


dc = DecisionTreeClassifier(ccp_alpha=0.0195)
dc.fit(X_train,y_train)
plt.figure(figsize=(15,7))
tree.plot_tree(dc,filled=True,class_names=['Not_placed','Placed'],feature_names=train.columns)


# # HyperParameter Tuning 

# In[71]:


params = {
    'RandomForest':{
        'model': RandomForestClassifier(),
        'params':{
            'n_estimators': [int(x) for x in np.linspace(start=1,stop=1200,num=10)],
            'max_depth':[int(x) for x in np.linspace(start=1,stop=30,num=5)],
            'min_samples_split':[2,5,10,12],
            'min_samples_leaf':[2,5,10,12],
            'max_features':['auto','sqrt'],
            'ccp_alpha':[0.0185,0.0190,0.0195,0.0200],
        }
    },
    
    'logistic':{
        'model':LogisticRegression(),
        'params':{
            'penalty':['l1', 'l2', 'elasticnet'],
            'C':[0.25,0.50,0.75,1.0],
            'tol':[1e-10,1e-5,1e-4,1e-3,0.025,0.25,0.50],
            'solver':['lbfgs','liblinear','saga','newton-cg'],
            'multi_class':['auto', 'ovr', 'multinomial'],
            'max_iter':[int(x) for x in np.linspace(start=1,stop=250,num=10)],
        }
    },
    'D-tree':{
        'model':DecisionTreeClassifier(),
        'params':{
            'criterion':['gini','entropy'],
            'splitter':['best','random'],
            'min_samples_split':[1,2,5,10,12],
            'min_samples_leaf':[1,2,5,10,12],
            'max_features':['auto','sqrt'],
            'ccp_alpha':[0.0185,0.0190,0.0195,0.0200],
        }
    },
    'SVM':{
        'model':SVC(),
        'params':{
            'C':[0.25,0.50,0.75,1.0],
            'tol':[1e-10,1e-5,1e-4,0.025,0.50,0.75],
            'kernel':['linear','poly','sigmoid','rbf'],
            'max_iter':[int(x) for x in np.linspace(start=1,stop=250,num=10)],
        }
    }
}


# In[72]:


scores = []
for model_name,mp in params.items():
    
    clf = RandomizedSearchCV(mp['model'],param_distributions=mp['params'],cv=5,n_iter=10,n_jobs=-1,scoring='accuracy')
    clf.fit(X_train,y_train)
    scores.append({
        'model_name':model_name,
        'best_score':clf.best_score_,
        'best_estimator':clf.best_estimator_
    })


# In[73]:


scores_df = pd.DataFrame(data = scores,columns = ['model_name','best_score','best_estimator'])
scores_df.head()


# In[74]:


scores


# In[75]:


# random forest model


rf = RandomForestClassifier(ccp_alpha=0.02, max_depth=30, max_features='sqrt',
                         min_samples_leaf=2, min_samples_split=12,
                         n_estimators=267)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))


# In[76]:


# logistic regression

lr = LogisticRegression(C=0.25, max_iter=111, multi_class='ovr', solver='newton-cg',
                     tol=1e-05)

lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))


# In[77]:


# decision tree

dc = DecisionTreeClassifier(ccp_alpha=0.019, criterion='entropy',
                         max_features='auto', min_samples_leaf=5,
                         min_samples_split=5, splitter='random')
dc.fit(X_train,y_train)
y_pred = dc.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))


# In[78]:


print(metrics.classification_report(y_test,rf.predict(X_test)))


# In[79]:


cn = metrics.confusion_matrix(y_test,rf.predict(X_test))
sn.heatmap(cn,annot=True,cmap='plasma')


# # Model Testing 

# In[80]:


traindata = np.array(train)
predicted = []
for i in range(len(traindata)):
    predicted.append(rf.predict([traindata[i]]))
    
predicted


# In[81]:


resultdf = train.copy()
resultdf['Actual'] = np.array(test)
resultdf['Predicted'] = np.array(predicted)
resultdf.head()


# In[82]:


resultdf['Actual'].value_counts()


# In[83]:


resultdf['Predicted'].value_counts()


# In[84]:


getmappings


# # Predicted Probability

# In[85]:


traindata = np.array(train)
predicted = []
for i in range(len(traindata)):
    predicted.append(rf.predict_proba([traindata[i]]))
    
predicted


# In[86]:


predicted[0][0],predicted[0][0][0],predicted[0][0][1]


# In[87]:


resultdf['Prob_not_getting_placed'] = np.array([predicted[i][0][0] for i in range(len(predicted))])
resultdf['Prob_getting_placed'] = np.array([predicted[i][0][1] for i in range(len(predicted))])
resultdf


# In[88]:


# saving the model

import pickle
file = open('campusplacementpredictor.pkl','wb')
pickle.dump(rf,file)
file.close()

