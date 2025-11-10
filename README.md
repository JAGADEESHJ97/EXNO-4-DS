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
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data

 ```
<img width="1394" height="410" alt="Screenshot 2025-10-07 111920" src="https://github.com/user-attachments/assets/a12642c5-bdfd-4e79-907b-d2e71edcaa6c" />

```
data.isnull().sum()

```
<img width="269" height="487" alt="Screenshot 2025-10-07 112010" src="https://github.com/user-attachments/assets/4e7ca11f-e232-4aa7-8210-9b4b87254789" />

```
missing=data[data.isnull().any(axis=1)]
missing

```
<img width="1358" height="421" alt="Screenshot 2025-10-07 112104" src="https://github.com/user-attachments/assets/a2514fa5-47f3-4e4f-a050-29fb21b8c55e" />

```
data2=data.dropna(axis=0)
data2

```
<img width="1394" height="428" alt="Screenshot 2025-10-07 112201" src="https://github.com/user-attachments/assets/0753e05a-795b-4edf-ae47-a0a31edee1ad" />

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

```
<img width="643" height="219" alt="Screenshot 2025-10-07 112317" src="https://github.com/user-attachments/assets/67492644-a44c-4b68-8152-d4c57cd6caa7" />

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs

```
<img width="365" height="426" alt="Screenshot 2025-10-07 112413" src="https://github.com/user-attachments/assets/6c857003-84a5-4c0a-858a-248386802859" />

```
data2

```
<img width="1266" height="418" alt="Screenshot 2025-10-07 112539" src="https://github.com/user-attachments/assets/a36442a9-e398-4865-8750-feda5019f2b1" />

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data

```
<img width="1494" height="469" alt="Screenshot 2025-10-07 112646" src="https://github.com/user-attachments/assets/2fb8f62d-50d6-434f-801c-5262f1da8478" />

```
columns_list=list(new_data.columns)
print(columns_list)

```
<img width="1480" height="37" alt="Screenshot 2025-10-07 112742" src="https://github.com/user-attachments/assets/5fa3c469-f628-4a3c-92d3-58ed2328584a" />

```
features=list(set(columns_list)-set(['SalStat']))
print(features)

```
<img width="835" height="36" alt="Screenshot 2025-10-07 112858" src="https://github.com/user-attachments/assets/a3e65bd5-2fd7-4d34-8091-3c164cd55663" />

```
y=new_data['SalStat'].values
print(y)

```
<img width="193" height="33" alt="Screenshot 2025-10-07 112952" src="https://github.com/user-attachments/assets/63cfa57b-d67e-495a-801b-2fe7357a4a1a" />

```
x=new_data[features].values
print(x)

```
<img width="362" height="142" alt="Screenshot 2025-10-07 113044" src="https://github.com/user-attachments/assets/dc843344-99d3-4eef-ac8d-3338934730d6" />

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)

```
<img width="336" height="68" alt="Screenshot 2025-10-07 113127" src="https://github.com/user-attachments/assets/e8d195cb-df9b-48e1-a05c-2fab986a574b" />

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)

```
<img width="146" height="42" alt="Screenshot 2025-10-07 113208" src="https://github.com/user-attachments/assets/71f5a273-e1ea-467f-a392-ca41820e47d4" />

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

```
<img width="181" height="33" alt="Screenshot 2025-10-07 113248" src="https://github.com/user-attachments/assets/16f06727-d3f5-4163-9ca2-04aa857c9c0c" />

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())

```

<img width="294" height="22" alt="Screenshot 2025-10-07 113332" src="https://github.com/user-attachments/assets/1d6d9e23-d7d0-45c2-b594-f5cef10e76fa" />

```
data.shape

```
<img width="134" height="29" alt="Screenshot 2025-10-07 113415" src="https://github.com/user-attachments/assets/5a5c169a-1fdf-4df9-80d8-94a3cfe6f3ab" />

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

```
<img width="400" height="35" alt="Screenshot 2025-10-07 113500" src="https://github.com/user-attachments/assets/84802d9e-a242-4d6c-b5b9-5e59f3d203ca" />

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()

```
<img width="456" height="206" alt="Screenshot 2025-10-07 113539" src="https://github.com/user-attachments/assets/f39159f7-7409-4be8-96e1-195ec97b9d33" />

```
tips.time.unique()

```
<img width="431" height="52" alt="Screenshot 2025-10-07 113632" src="https://github.com/user-attachments/assets/005aae6e-cec1-4c57-a77a-499940653d1f" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)

```
<img width="208" height="75" alt="Screenshot 2025-10-07 113720" src="https://github.com/user-attachments/assets/fbc57922-6833-4ad4-9705-b771c78f29fe" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")

```
<img width="399" height="53" alt="Screenshot 2025-10-07 113818" src="https://github.com/user-attachments/assets/a8bf7988-7d27-441e-81fd-1014db242175" />





















# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and
save the data to a file is been executed.
