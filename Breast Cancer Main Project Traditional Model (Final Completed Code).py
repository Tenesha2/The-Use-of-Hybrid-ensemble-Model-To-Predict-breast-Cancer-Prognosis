#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


#Load the dataset
Data = pd.read_csv('/Users/teneshasmith/Downloads/Breast_Cancer.csv')


# In[3]:


Data

Setting Up The Dataset
# In[4]:


#Relevant Packages
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[5]:


#Extracting input and output data
X = Data.drop(['Status', 'Survival Months'], axis=1)#all columns EXCEPT Status and Survival
y=Data.iloc[:,15] #Survival is our target feature hence column 14


# In[6]:


#confirming columns are correctly allocated
X


# In[7]:


y

Normalising and Encoding The Dataset
# In[8]:


# Specify indices for ordinal and nominal variables
ordinal_indices = [ 3, 4, 5, 6, 7, 8]  # Indices of ordinal variables
nominal_indices = [1,2,10,11]  # Add indices for nominal variables if needed

# Defining the ColumnTransformer with separate handling for ordinal and nominal variables
ct = ColumnTransformer(
    transformers=[
        ("ordinal", OrdinalEncoder(), ordinal_indices),
        ("nominal", OneHotEncoder(), nominal_indices)  # Add this transformer if nominal variables are present
    ],
    remainder='passthrough'  # Keep other columns unchanged
)

# Apply the transformations to your dataset
X_transformer= ct.fit_transform(X)


# In[9]:


#encoding the output data (Survival)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(y)

#checking columns have been correctly encoded
y


# In[11]:


# Ensure X_transformer is dense before normalization
if hasattr(X_transformer, "toarray"):  # Check if it's sparse
    X_transformer = X_transformer.toarray()

# Normalize the transformed data
scaler = MinMaxScaler()
X = scaler.fit_transform(X_transformer)


print("\nNormalized Data:")
print(X)

Splitting the Dataset Into Training and Testing
# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

Model Packages
# In[13]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn import svm

Metric Packages (visual)
# In[14]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

Metric Packages (values)
# In[15]:


def confusion_metrics (conf_matrix):

    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    print('True Positives:', TP)
    print('True Negatives:', TN)
    print('False Positives:', FP)
    print('False Negatives:', FN)
## Evaluation metrics ##
# calculate accuracy
    conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))
# calculate mis-classification
    conf_misclassification = 1- conf_accuracy
 # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))
# calculate the specificity
    conf_specificity = (TN / float(TN + FP))
# calculate precision
    conf_precision = (TP / float(TP + FP))
# calculate f_1 score//calculating the score and printing the results
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
    print('-'*50)
    print(f'Accuracy: {round(conf_accuracy,2)}') 
    print(f'Mis-Classification: {round(conf_misclassification,2)}') 
    print(f'Sensitivity: {round(conf_sensitivity,2)}') 
    print(f'Specificity: {round(conf_specificity,2)}') 
    print(f'Precision: {round(conf_precision,2)}')
    print(f'f_1 Score: {round(conf_f1,2)}')

Understanding The Allocation of 0 and 1
# In[16]:


print("Unique classes in target variable:")
print(Data['Status'].unique())  

# Map the classes to model predictions
print("\nClass mapping in the model:")
from sklearn.preprocessing import LabelEncoder


label_encoder = LabelEncoder()
label_encoder.fit(Data['Status'])  
print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

Support Vector Machine
# In[17]:


#Defining the classification type
SVM = svm.SVC(probability=True)
#Training the model
SVM.fit(X_train, y_train)
#Predict with model
y_pred1=SVM.predict(X_test)


# In[18]:


#Printing the models predictions
print(y_pred1[:4], "...")


# In[19]:


#confusion matrix//code applies labels and specifies the design of the CF//based on prediction results
cm1 = confusion_matrix(y_test, y_pred1, labels=SVM.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm1,display_labels=SVM.classes_)
#display the plot and add a title
disp.plot()
plt.title("Support Vector Machine")


# In[20]:


#printing the evaluation metrics for SVM
print('SVM metrics\n')
confusion_metrics(cm1)
print('\n\n')

Logistic Regression
# In[17]:


#defining the model
logre = LogisticRegression()
#fitting the model
logre.fit(X_train, y_train)
#Predict with model
y_pred2=logre.predict(X_test)

#Printing the models predictions
print(y_pred2[:4], "...")


# In[18]:


#confusion matrix//code applies labels and specifies the design of the CF//based on prediction results
cm2 = confusion_matrix(y_test, y_pred2, labels=logre.classes_)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2,display_labels=logre.classes_)
#display the plot and add a title
disp2.plot()
plt.title("Logistic Regression")


# In[19]:


#printing the evaluation metrics for Logistic Regression
print('Logistic Regression metrics\n')
confusion_metrics(cm2)
print('\n\n')

K-nearest Neighbours
# In[23]:


#Defining the model
KNN = KNeighborsClassifier()
#training the models
KNN.fit(X_train, y_train)
#predicting with the model
y_pred3=KNN.predict(X_test)
#Printing the models predictions
print(y_pred2[:4], "...")


# In[24]:


#Confusion matrix:KNN //code applies labels and specifies the design of the CF//based on prediction results
cm3 = confusion_matrix(y_test, y_pred3, labels=KNN.classes_)
#specifying the design
disp3 = ConfusionMatrixDisplay(confusion_matrix=cm3,display_labels=KNN.classes_)
#generates the plot
disp3.plot()
#displays the title
plt.title("K-Nearest Neighbours")


# In[25]:


#Evaluations based on confusion matrix
print('KNN metrics\n')
confusion_metrics(cm3)
print('\n\n')

Decision Trees
# In[26]:


#Defining the model
DT=DecisionTreeClassifier()
#Training the model
DT.fit(X_train, y_train)
#predicting with the model
y_pred4=DT.predict(X_test)


# In[27]:


cm4 = confusion_matrix(y_test, y_pred4, labels=DT.classes_)
#determines the display
disp = ConfusionMatrixDisplay(confusion_matrix=cm4,display_labels=DT.classes_)
#to generate the plot
disp.plot()
#to set the title
plt.title("Decision Tree")


# In[28]:


#evaluation metrics based on the prior confusion matrix
print('DT metrics\n')
confusion_metrics(cm4)
print('\n\n')


# In[29]:


#Plotting the features to determine importance, on predicting income
#specifiying the features
features = Data.columns
#specifying the importances
importances = DT.feature_importances_
#specifying the indices
indices = np.argsort(importances)
#creates the title
plt.title('Feature Importances For Decision Tree')
#defining the colour and alignment
plt.barh(range(len(indices)), importances[indices], color='r', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
#defining the labels on the x axis
plt.xlabel('Relative Importance')
#to show the plot
plt.show()

#confirming the top 3 features 
DT_top_features = sorted(zip(importances, features), reverse=True) [:3]
print(DT_top_features)

Random Forest
# In[30]:


#defining the model
RF = RandomForestClassifier()
#Training the model
RF.fit(X_train, y_train)
#Predicting with the model
y_pred5=RF.predict(X_test)


# In[31]:


#creating the confusion matrix
cm5 = confusion_matrix(y_test, y_pred5, labels=RF.classes_)
#specifying the design
disp = ConfusionMatrixDisplay(confusion_matrix=cm5,display_labels=RF.classes_)
#to show the plot
disp.plot()
#to set the title
plt.title("Random Forest")


# In[32]:


#evaluation based on the confusion matrix
print('RF metrics\n')
confusion_metrics(cm5)
print('\n\n')


# In[33]:


# Viewing importance of features to help predict income
#specifies features
features = Data.columns
#specifies the importances
importances = RF.feature_importances_
#specifies the indices
indices = np.argsort(importances)
#designs the plot#
#title
plt.title('Feature Importances For Random Forest') 
#alignment and colour scheme
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
#defines label on the x axis
plt.xlabel('Relative Importance')
#shows the plot
plt.show()

#confirming the top 3 features 
RF_top_features = sorted(zip(importances, features), reverse=True) [:3]
print(RF_top_features)

Gussian Naive-Bayes
# In[34]:


gnb = GaussianNB()
gnb.fit(X_train, y_train)
#Predicting with the model
y_pred6=gnb.predict(X_test)


# In[35]:


#creating the confusion matrix
cm6 = confusion_matrix(y_test, y_pred6, labels=gnb.classes_)
#specifying the design
disp = ConfusionMatrixDisplay(confusion_matrix=cm6,display_labels=gnb.classes_)
#to show the plot
disp.plot()
#to set the title
plt.title("Gussian Naive-Bayes")


# In[36]:


#evaluation based on the confusion matrix
print(' gnb.metrics\n')
confusion_metrics(cm6)
print('\n\n')


# # Explainability

# In[20]:


import shap
import warnings
# Smaller sample of the test set to calculate SHAP values
X_test_sample = shap.sample(X_test, 100)
# Suppress warnings from sklearn's least_angle module
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.linear_model._least_angle")

Random Forest
# In[38]:


# Initialize TreeExplainer
explainer = shap.TreeExplainer(RF)

# Compute SHAP values
shap_values = explainer.shap_values(X_test_sample)
        
# Extract SHAP values for the died class
shap_values_survival = shap_values[..., 1]

# Visualize feature contributions for died
shap.summary_plot(shap_values_survival, X_test_sample,feature_names=Data.columns, max_display=15)

# Extract SHAP values for the survival class
shap_values_died = shap_values[..., 0]

# Visualize feature contributions for survival
shap.summary_plot(shap_values_died, X_test_sample, feature_names=Data.columns,max_display=15)

Decision Trees
# In[39]:


# Initialize TreeExplainer
explainer1 = shap.TreeExplainer(DT)

# Compute SHAP values
shap_values = explainer1.shap_values(X_test_sample)

# Extract SHAP values for the died class
shap_values_survival1 = shap_values[..., 1]

# Visualize feature contributions for died
shap.summary_plot(shap_values_survival1, X_test_sample,feature_names=Data.columns, max_display=15)

# Extract SHAP values for the survival class
shap_values_died2 = shap_values[..., 0]

# Visualize feature contributions for survival
shap.summary_plot(shap_values_died2, X_test_sample,feature_names=Data.columns, max_display=15)

Logistic Regression
# In[24]:


explainer_linear = shap.LinearExplainer(logre, X_train)
shap_values_linear = explainer_linear.shap_values(X_test)


# In[25]:


#alive
shap.summary_plot(shap_values_linear, X_test, feature_names=Data.columns)


# In[26]:


# Dead
shap_values_dead3 = -shap_values_linear

# Visualize feature contributions for predicting "dead"
shap.summary_plot(shap_values_dead3, X_test, feature_names=Data.columns, max_display=15)

Support vector Machine
# In[22]:


explainer_kernel2 = shap.KernelExplainer(SVM.predict, X_train[:100]) 
shap_values_kernel2 = explainer_kernel2.shap_values(X_test_sample)


# In[24]:


#alive
shap.summary_plot(shap_values_kernel2, X_test_sample, feature_names=Data.columns)


# In[26]:


# Dead
shap_values_dead2 = -shap_values_kernel2

# Visualize feature contributions for predicting "dead"
shap.summary_plot(shap_values_dead2, X_test_sample, feature_names=Data.columns, max_display=15)

KNN
# In[69]:


explainer_kernel1 = shap.KernelExplainer(KNN.predict, X_train[:100])  
shap_values_kernel1 = explainer_kernel1.shap_values(X_test_sample)


# In[70]:


#alive
shap.summary_plot(shap_values_kernel1, X_test_sample, feature_names=Data.columns)


# In[71]:


# Dead
shap_values_dead = -shap_values_kernel1

# Visualize feature contributions for predicting "dead"
shap.summary_plot(shap_values_dead, X_test_sample, feature_names=Data.columns, max_display=15)

NB
# In[73]:


explainer_kernel = shap.KernelExplainer(gnb.predict, X_train[:100])  
shap_values_kernel = explainer_kernel.shap_values(X_test_sample)


# In[75]:


#alive
shap.summary_plot(shap_values_kernel, X_test_sample, feature_names=Data.columns)


# In[76]:


# Dead
shap_values_dead1 = -shap_values_kernel

# Visualize feature contributions for predicting "dead"
shap.summary_plot(shap_values_dead1, X_test_sample, feature_names=Data.columns, max_display=15)

