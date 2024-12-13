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
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
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

# Define the ColumnTransformer with separate handling for ordinal and nominal variables
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


# In[10]:


#normalising the input data
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


from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

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

Forming Hybrid-Model 1
# In[16]:


# get a stacking ensemble of models
def get_stacking():
   level0 = list()
   level0.append(('lr',  LogisticRegression()))
   level0.append(('svm', svm.SVC(probability=True)))  
   level0.append(('nb', GaussianNB()))
 # define meta learner model
   level1 = RandomForestClassifier()
 # define the stacking ensemble
   model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
   return model

model = get_stacking()


# In[17]:


# Train the stacking ensemble
model.fit(X_train, y_train)


# In[18]:


# Step 7: Make predictions on the test set
y_pred = model.predict(X_test)


# In[19]:


print(y_pred[:4], "...")


# In[20]:


#confusion matrix//code applies labels and specifies the design of the CF//based on prediction results
cm1 = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm1,display_labels=model.classes_)
#display the plot and add a title
disp.plot()
plt.title("Hybrid-Ensemble Model 1")


# In[21]:


#printing the evaluation metrics for the Hybrid Model
print('model metrics\n')
confusion_metrics(cm1)
print('\n\n')


# Model 2

# In[22]:


# get a stacking ensemble of models
def get_stacking():
   level0 = list()
   level0.append(('rf',  RandomForestClassifier()))
   level0.append(('svm', svm.SVC(probability=True)))  
   level0.append(('nb', GaussianNB()))
 # define meta learner model
   level1 = LogisticRegression()
 # define the stacking ensemble
   model2 = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
   return model2

model2 = get_stacking()


# In[23]:


# Train the stacking ensemble
model2.fit(X_train, y_train)


# In[24]:


# Step 7: Make predictions on the test set
y_pred2 = model2.predict(X_test)
print(y_pred2[:4], "...")


# In[25]:


#confusion matrix//code applies labels and specifies the design of the CF//based on prediction results
cm2 = confusion_matrix(y_test, y_pred2, labels=model2.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm2,display_labels=model2.classes_)
#display the plot and add a title
disp.plot()
plt.title("Hybrid-Ensemble Model 2")


# In[26]:


#printing the evaluation metrics for the Hybrid Model
print('model2 metrics\n')
confusion_metrics(cm2)
print('\n\n')

Hybrid Model 3
# In[27]:


# get a stacking ensemble of models
def get_stacking():
   level0 = list()
   level0.append(('rf',  RandomForestClassifier()))
   level0.append(('lr', LogisticRegression()))  
   level0.append(('nb', GaussianNB()))
 # define meta learner model
   level1 = svm.SVC(probability=True)
 # define the stacking ensemble
   model3 = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
   return model3

model3 = get_stacking()


# In[28]:


# Train the stacking ensemble
model3.fit(X_train, y_train)


# In[29]:


# Step 7: Make predictions on the test set
y_pred3 = model3.predict(X_test)
print(y_pred3[:4], "...")


# In[30]:


#confusion matrix//code applies labels and specifies the design of the CF//based on prediction results
cm3 = confusion_matrix(y_test, y_pred3, labels=model3.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm3,display_labels=model3.classes_)
#display the plot and add a title
disp.plot()
plt.title("Hybrid-Ensemble Model 3")


# In[31]:


#printing the evaluation metrics for the Hybrid Model
print('model3 metrics\n')
confusion_metrics(cm3)
print('\n\n')

Model 4
# In[32]:


# get a stacking ensemble of models
def get_stacking():
   level0 = list()
   level0.append(('rf',  RandomForestClassifier()))
   level0.append(('lr', LogisticRegression()))  
   level0.append(('svm', svm.SVC(probability=True)))
 # define meta learner model
   level1 = GaussianNB()
 # define the stacking ensemble
   model4 = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
   return model4

model4 = get_stacking()


# In[33]:


# Train the stacking ensemble
model4.fit(X_train, y_train)


# In[34]:


# Step 7: Make predictions on the test set
y_pred4 = model4.predict(X_test)
print(y_pred4[:4], "...")


# In[35]:


#confusion matrix//code applies labels and specifies the design of the CF//based on prediction results
cm4 = confusion_matrix(y_test, y_pred4, labels=model4.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm4,display_labels=model4.classes_)
#display the plot and add a title
disp.plot()
plt.title("Hybrid-Ensemble Model 4")


# In[36]:


#printing the evaluation metrics for the Hybrid Model
print('model4 metrics\n')
confusion_metrics(cm4)
print('\n\n')


# # Explainability

# In[37]:


#packages
import shap
# Limit the explanation to 100 samples of the test data
X_test_sample = shap.sample(X_test, 100)

Model 1
# In[ ]:


explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
shap_values = explainer.shap_values(X_test_sample)


# In[40]:


# Extract SHAP values for the died class
shap_values_survival = shap_values[..., 1]

# Visualize feature contributions for died
shap.summary_plot(shap_values_survival, X_test_sample,feature_names=Data.columns, max_display=15)

# Extract SHAP values for the survived class
shap_values_died = shap_values[..., 0]

# Visualize feature contributions for survived
shap.summary_plot(shap_values_died, X_test_sample,feature_names=Data.columns, max_display=15)


# Model 2

# In[41]:


explainer1 = shap.KernelExplainer(model2.predict_proba, X_train[:100])
shap_values1 = explainer1.shap_values(X_test_sample)


# In[43]:


# Extract SHAP values for the died class
shap_values_survival1 = shap_values1[..., 1]

# Visualize feature contributions for died
shap.summary_plot(shap_values_survival1, X_test_sample,feature_names=Data.columns, max_display=15)

# Extract SHAP values for the alive class
shap_values_died2 = shap_values1[..., 0]

# Visualize feature contributions for alive
shap.summary_plot(shap_values_died2, X_test_sample,feature_names=Data.columns, max_display=15)

Model 3
# In[44]:


explainer2 = shap.KernelExplainer(model3.predict_proba, X_train[:100])
shap_values2 = explainer2.shap_values(X_test_sample)


# In[45]:


# Extract SHAP values for the died class
shap_values_survival3 = shap_values2[..., 1]

# Visualize feature contributions for died
shap.summary_plot(shap_values_survival3, X_test_sample,feature_names=Data.columns, max_display=15)

# Extract SHAP values for the survival class
shap_values_died4 = shap_values2[..., 0]

# Visualize feature contributions for survival
shap.summary_plot(shap_values_died4, X_test_sample,feature_names=Data.columns, max_display=15)

Model 4
# In[46]:


explainer3 = shap.KernelExplainer(model4.predict_proba, X_train[:100])
shap_values3 = explainer3.shap_values(X_test_sample)


# In[47]:


# Extract SHAP values for the died class
shap_values_survival5 = shap_values3[..., 1]

# Visualize feature contributions for died
shap.summary_plot(shap_values_survival5, X_test_sample,feature_names=Data.columns, max_display=15)

# Extract SHAP values for the survival class
shap_values_died6 = shap_values3[..., 0]

# Visualize feature contributions for survival
shap.summary_plot(shap_values_died6, X_test_sample,feature_names=Data.columns, max_display=15)


# In[ ]:




