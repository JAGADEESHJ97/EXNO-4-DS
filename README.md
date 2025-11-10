# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1) (1).csv",na_values=[ " ?"])
data
```

<img width="1647" height="723" alt="image" src="https://github.com/user-attachments/assets/06756d22-d1ac-404c-ba3c-1cc1718ea801" />


```
data.isnull().sum()
```

<img width="253" height="605" alt="image" src="https://github.com/user-attachments/assets/46db43dc-6140-4902-8ae9-3324002fa2a5" />

```
missing=data[data.isnull().any(axis=1)]
missing
```

<img width="1629" height="692" alt="image" src="https://github.com/user-attachments/assets/e121ee6a-73df-4fb7-b891-c91e0d54bfad" />


```
data2=data.dropna(axis=0)
data2
```

<img width="1630" height="717" alt="image" src="https://github.com/user-attachments/assets/7050d1cc-7eb2-4553-af94-620874ae4c5e" />


```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

<img width="1445" height="401" alt="image" src="https://github.com/user-attachments/assets/de952ba6-bd4b-4a77-b17f-f31d4e3d69e0" />


```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```


<img width="410" height="523" alt="image" src="https://github.com/user-attachments/assets/77d21bb2-e1aa-4a36-9e2c-dd3a2796ab35" />



```
data2
```

<img width="1550" height="497" alt="image" src="https://github.com/user-attachments/assets/57dfe7fe-fb92-4891-86ce-abaf04813ee6" />



```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

<img width="1676" height="579" alt="image" src="https://github.com/user-attachments/assets/96601383-ba7f-4d8b-bb8e-4efa9d76bfe7" />


```
columns_list=list(new_data.columns)
print(columns_list)
```

<img width="1679" height="41" alt="image" src="https://github.com/user-attachments/assets/5843d7a1-7632-4c73-a779-79ebfa9635c4" />



```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

<img width="1668" height="40" alt="image" src="https://github.com/user-attachments/assets/ce6c6cd8-4b2a-4873-a398-b83fb6957a72" />


```
y=new_data['SalStat'].values
print(y)

```

<img width="203" height="36" alt="image" src="https://github.com/user-attachments/assets/92c0f1bd-10b0-459b-babc-1d28a2a6a081" />


```
x=new_data[features].values
print(x)
```

<img width="387" height="165" alt="image" src="https://github.com/user-attachments/assets/532cfb0c-2816-4f2e-b5cc-2185ac5d1c9b" />


```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```

<img width="312" height="81" alt="image" src="https://github.com/user-attachments/assets/352e6773-3a67-4378-86ce-610ca7e72e42" />


```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

<img width="159" height="53" alt="image" src="https://github.com/user-attachments/assets/60e44650-98d6-4f2c-b74e-98e44ced0941" />


```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```

<img width="204" height="29" alt="image" src="https://github.com/user-attachments/assets/4f59f53d-2bfc-45a7-9d27-d6ff78d20e4a" />


```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```

<img width="296" height="31" alt="image" src="https://github.com/user-attachments/assets/6ef453c2-403b-4cc4-960b-f1ba3be169d8" />


```
data.shape
```

<img width="123" height="29" alt="image" src="https://github.com/user-attachments/assets/9fbf9113-1ec4-4528-a5c6-84f8546a475e" />


```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
'Feature1': [1,2,3,4,5],
'Feature2': ['A','B','C','A','B'],
'Feature3': [0,1,1,0,1],
'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

<img width="1674" height="348" alt="image" src="https://github.com/user-attachments/assets/ddc8acb5-ac43-44a0-a231-c4b7ff94e325" />



```
tips.time.unique()
```
<img width="491" height="61" alt="image" src="https://github.com/user-attachments/assets/5252b69e-6b54-4bd5-a521-7c52ffbedacc" />


```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

<img width="264" height="100" alt="image" src="https://github.com/user-attachments/assets/ffff4746-5f29-4eae-ba0b-aad054a41a91" />


```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

<img width="464" height="56" alt="image" src="https://github.com/user-attachments/assets/cc9dc94e-b6fe-4602-bc40-36c76e4a3212" />


# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and
save the data to a file is been executed.

