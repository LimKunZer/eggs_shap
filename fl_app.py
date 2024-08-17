#import libraries
import streamlit as st
import shap
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
from sklearn.metrics import classification_report

#import data files
df = pd.read_csv('foodwaste.csv')

#drop unnecesarry columns from the data frame
df.drop(columns=['Domain', 'Unit', 'Flag', 'Flag Description'], inplace=True)

df['Item'] = df['Item'].replace({ "Eggs from other birds in shell, fresh, n.e.c.": "Other eggs"})
df['Item'] = df['Item'].replace({ "Hen eggs in shell, fresh": "Chicken eggs"})
df['Element'] = df['Element'].replace({ "Export Quantity": "Export"})

# replace null values with appropriate values, then display the import quantity of non-chicken eggs for Malaysia & the production of non-chicken eggs in Ghana
area = ""
element = ""
item = ""
yearsCount = 0
nullYearsCount = 0
relatedValuesSum = 0
for index, row in df.iterrows():
  if(area == row['Area'] and element == row['Element'] and item == row['Item']):
    yearsCount += 1
    if (pd.isnull(row['Value'])):
      nullYearsCount += 1
    else:
      relatedValuesSum += row['Value']
  else:
    yearsCount = 1
    nullYearsCount = 1
    relatedValuesSum = 0
  if(index % 10 == 9):
    for j in range(index, index - 10, -1):
      if (nullYearsCount == 10 and pd.isnull(df.at[j, 'Value'])):
        df.at[j, 'Value'] = 0
      elif (nullYearsCount != 10 and pd.isnull(df.at[j, 'Value'])):
        df.at[j, 'Value'] = relatedValuesSum / (10 - (nullYearsCount - 1))
  area = row['Area']
  element = row['Element']
  item = row['Item']

# Rename the "Area" column
df.rename(columns={"Area": "Country"} , inplace=True)

# refactor the data for the model to work
df['Country'] = df['Country'].replace({ "Malaysia": "1", "Peru":"2", "Ghana":"3"})
df['Item'] = df['Item'].replace({ "Chicken eggs": "1", "Other eggs":"2"})

# split the data to training set & test set
from sklearn.model_selection import train_test_split
X = df.drop(["Element"], axis=1)
y = df["Element"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 36)

# fit extra trees classifier model to data
from sklearn.ensemble import ExtraTreesClassifier

et_clf = ExtraTreesClassifier(random_state = 42) #classification model
et_clf.fit(X_train, y_train)

# calculate accuracy, precision, recall & f1-score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
y_pred = et_clf.predict(X_test)

# SHAP explainer
fl_explainer = shap.TreeExplainer(et_clf)
fl_shap_values = fl_explainer.shap_values(X_test)

# Streamlit app
st.title("SHAP Analysis")

# General SHAP Analysis
st.header("General SHAP Analysis")
st.dataframe(classification_report(y_pred, y_test,output_dict=True))

# Summary plot
st.subheader("Summary Plot")
fig, ax = plt.subplots()
shap.summary_plot(fl_shap_values, X_test, show=False)
st.pyplot(fig)

# Force plot
st.subheader("Force Plot")
st_shap(shap.plots.force(fl_explainer.expected_value[1], fl_shap_values[1][6,:], X_test.iloc[6, :], matplotlib = True), height=400, width=1000)

# Decision plot
st.subheader("Decision Plot")
st_shap(shap.decision_plot(fl_explainer.expected_value[0], fl_shap_values[0], X_test.columns))
