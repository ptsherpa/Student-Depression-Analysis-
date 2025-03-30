#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[3]:


data = pd.read_csv('C:\\Users\\Desktop\\DataSet\\student_depression_dataset.csv', encoding='ISO-8859-1')


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


# Check for missing values
data.isnull().sum()


# In[7]:


# Check data types
data.dtypes


# In[8]:


data.shape


# In[9]:


# Using drop method
data = data.drop("Work Pressure", axis=1)
data = data.drop("Job Satisfaction", axis=1)


# In[10]:


data = data.drop("id", axis=1)


# In[11]:


# Using drop method
data = data.drop("City", axis=1)
data = data.drop("Profession", axis=1)


# In[12]:


data.head()


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt

# Count the total number of males and females
total_males = data[data["Gender"] == "Male"]["Gender"].count()
total_females = data[data["Gender"] == "Female"]["Gender"].count()

# Visualize the gender distribution
plt.figure(figsize=(10, 6))
# Customize autopct to show both percentage and count
plt.pie(
    [total_males, total_females], 
    labels=["Males", "Females"], 
    autopct=lambda p: f'{p:.1f}%\n({int(p * (total_males + total_females) / 100)})'
)
plt.title("Gender Distribution")
plt.show()


# In[14]:


#age distribution
sns.countplot(x="Age",data= data)
plt.rcParams["figure.figsize"] = (20,5)


# In[15]:


# Count the total number of Depression
Depression = data[data["Depression"] == 1]["Depression"].count()
Not_Depression= data[data["Depression"] == 1]["Depression"].count()

plt.figure(figsize=(10, 6))
# Customize autopct to show both percentage and count
plt.pie(
    [Depression, Not_Depression], 
    labels=["Depression", "Not Depression"], 
    autopct=lambda p: f'{p:.1f}%\n({int(p * (Depression + Not_Depression) / 100)})'
)
plt.title("Depression Distribution")
plt.show()


# In[16]:


# Count the number of males and females who passed and failed

# Define the data
males_Depression = data[(data["Gender"] == "Male") & (data["Depression"] == 1)]["Gender"].count()
males_Not_Depression = data[(data["Gender"] == "Male") & (data["Depression"] == 0)]["Gender"].count()
females_Depression = data[(data["Gender"] == "Female") & (data["Depression"] == 1)]["Gender"].count()
females_Not_Depression = data[(data["Gender"] == "Female") & (data["Depression"] == 0)]["Gender"].count()
categories = ["Males (Depression)", "Males (Not Depression)", "Females (Depression)", "Females (Not Depression)"]
counts = [males_Depression, males_Not_Depression, females_Depression, females_Not_Depression]
colors = ["blue", "lightblue", "pink", "lightcoral"]

# Create the bar plot
plt.figure(figsize=(10, 6))
bars = plt.bar(categories, counts, color=colors)

# Add value annotations on top of the bars
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{int(bar.get_height())}', 
             ha='center', fontsize=12)

# Add labels, title, and grid
plt.title("Depression vs. Not Depression by Gender", fontsize=14)
plt.xlabel("Gender and Status", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.show()


# In[17]:


# Plot the age distribution for males and females
plt.figure(figsize=(10, 6))
plt.hist(data[data["Gender"] == "Male"]["Age"], bins=100, alpha=0.8, label="Males")
plt.hist(data[data["Gender"] == "Female"]["Age"], bins=100, alpha=0.8, label="Females")
plt.legend(loc='upper right')
plt.title("Age Distribution by Gender")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# In[18]:


# Distribution of Academic Pressure
sns.histplot(data['Academic Pressure'], kde=True)
plt.title('Distribution of Academic Pressure')
plt.show()


# In[19]:


import plotly.express as px

# Count the occurrences for each combination
suicidal_counts = data.groupby(["Have you ever had suicidal thoughts ?", "Depression"]).size().reset_index(name="Count")

# Plot using Plotly
fig = px.bar(
    suicidal_counts,
    x="Have you ever had suicidal thoughts ?",
    y="Count",
    color="Depression",
    barmode="group",
    title="Depression vs Suicidal Thoughts",
    labels={"Have you ever had suicidal thoughts ?": "Suicidal Thoughts", "Count": "Number of Individuals", "Depression": "Depression"},
    color_discrete_sequence=["steelblue", "lightcoral"]
)

# Show the interactive plot
fig.show()


# In[20]:



# Count the occurrences for each combination
suicidal_counts = data.groupby(["Have you ever had suicidal thoughts ?", "Depression"]).size().unstack()

# Plot the bar chart
suicidal_counts.plot(kind="bar", figsize=(10, 6), color=["lightcoral", "steelblue"], edgecolor="black")

# Customize the plot
plt.title("Depression vs Suicidal Thoughts", fontsize=14)
plt.xlabel("Suicidal Thoughts", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=0)  # Keep x-axis labels horizontal
plt.legend(title="Depression", labels=["No Depression", "Depression"])
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.show()


# In[21]:


# Count the occurrences for each combination
financial_counts = data.groupby(["Financial Stress", "Depression"]).size().unstack()


# Plot the bar chart
financial_counts.plot(kind="bar", figsize=(10, 6), color=["steelblue", "lightcoral"], edgecolor="black")

# Customize the plot
plt.title("Depression vs Financial Stress", fontsize=14)
plt.xlabel("Financial Stress Level (1.0 = Low, 5.0 = High)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=0)  # Keep x-axis labels horizontal
plt.legend(title="Depression", labels=["No Depression", "Depression"])
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.show()


# In[22]:



# Count the occurrences for each combination
financial_counts = data.groupby(["Financial Stress", "Depression"]).size().reset_index(name="Count")

# Plot using Plotly
fig = px.bar(
    financial_counts,
    x="Financial Stress",
    y="Count",
    color="Depression",
    barmode="group",
    title="Depression vs Financial Stress",
    labels={"Financial Stress": "Financial Stress Level (1.0 = Low, 5.0 = High)", "Count": "Number of Individuals", "Depression": "Depression"},
    color_discrete_sequence=["steelblue", "lightcoral"]
)

# Show the interactive plot
fig.show()


# In[23]:


# Count occurrences for each combination
family_history_counts = data.groupby(["Family History of Mental Illness", "Depression"]).size().unstack()

# Plot the stacked bar chart
family_history_counts.plot(kind="bar", stacked=True, figsize=(10, 6), color=["steelblue", "lightcoral"], edgecolor="black")

# Customize the plot
plt.title("Depression vs Family History of Mental Illness", fontsize=14)
plt.xlabel("Family History of Mental Illness", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=0)
plt.legend(title="Depression", labels=["No Depression", "Depression"])
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.show()


# In[24]:


# Display the unique values in the 'Work/Study Hours' column
unique_values = data['Work/Study Hours'].unique()
print("Unique values in 'Work/Study Hours':", unique_values)


# In[25]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(x="Depression", y="Work/Study Hours", data=data, palette="muted")

# Customize the plot
plt.title("Distribution of Work/Study Hours vs Depression", fontsize=14)
plt.xlabel("Depression (0 = No, 1 = Yes)", fontsize=12)
plt.ylabel("Work/Study Hours", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.show()


# In[26]:


# Count occurrences for each combination
degree_counts = data.groupby(["Degree", "Depression"]).size().unstack(fill_value=0)

# Create the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(degree_counts, annot=True, fmt="d", cmap="coolwarm", cbar=True)

# Customize the plot
plt.title("Heatmap of Degree vs Depression", fontsize=14)
plt.xlabel("Depression (0 = No, 1 = Yes)", fontsize=12)
plt.ylabel("Degree", fontsize=12)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

# Show the plot
plt.show()


# In[27]:


# Count occurrences for each combination
diet_counts = data.groupby(["Dietary Habits", "Depression"]).size().unstack(fill_value=0)

# Prepare data for the donut chart
diet_labels = diet_counts.index
depression_counts = diet_counts.sum(axis=1)  # Total individuals per dietary habit

# Create the pie chart with counts and percentages
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    depression_counts, 
    labels=diet_labels, 
    autopct=lambda pct: f"{int(round(pct/100.*depression_counts.sum()))}\n({pct:.1f}%)", 
    pctdistance=0.85, 
    colors=['lightblue', 'orange', 'lightgreen']
)

# Add a white circle to create the "donut"
plt.gca().add_artist(plt.Circle((0, 0), 0.70, color='white'))

# Customize the plot
plt.setp(autotexts, size=10, weight="bold")
plt.title("Dietary Habits Distribution with Counts and Percentages", fontsize=14)

# Show the plot
plt.show()


# In[28]:


# Group the data
diet_depression_counts = data.groupby(["Dietary Habits", "Depression"]).size().unstack(fill_value=0)

# Plot the data
ax = diet_depression_counts.plot(kind='barh', figsize=(12, 6), stacked=True, color=['skyblue', 'salmon'], edgecolor="black")

# Annotate counts on the bars
for i, bars in enumerate(ax.containers):
    for bar in bars:
        width = bar.get_width()
        if width > 0:
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{int(width)}', va='center')

# Customize the plot
plt.title("Dietary Habits vs Depression (with Counts)", fontsize=14)
plt.xlabel("Count", fontsize=12)
plt.ylabel("Dietary Habits", fontsize=12)
plt.legend(["No Depression", "Depression"], title="Depression", loc='upper right')
plt.grid(axis="x", linestyle="--", alpha=0.7)

# Show the plot
plt.show()


# In[29]:


data.head()


# In[30]:




# List of columns to convert
columns_to_convert = [
    "Age", "Academic Pressure", "CGPA", 
    "Study Satisfaction", 
    "Work/Study Hours", "Financial Stress"
]

# Step 1: Replace invalid string patterns with NaN
data[columns_to_convert] = data[columns_to_convert].apply(pd.to_numeric, errors='coerce')

# Step 2: Handle missing values (e.g., replace NaN with default values or mean)
data = data.fillna(0)  # Replace NaN with 0 (you can replace with mean or median instead)

# Step 3: Convert to integers
data[columns_to_convert] = data[columns_to_convert].astype(int)


# In[31]:


data.head()


# # Label Coded

# In[32]:


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()
# yes = 1, no=0

# Encode 'Have you ever had suicidal thoughts?'
data["Have you ever had suicidal thoughts ?"] = label_encoder.fit_transform(data["Have you ever had suicidal thoughts ?"])

# Encode 'Family History of Mental Illness'
data["Family History of Mental Illness"] = label_encoder.fit_transform(data["Family History of Mental Illness"])

#male=1, female =0
data["Gender"]=label_encoder.fit_transform(data["Gender"])


# In[33]:


data.head()


# In[34]:


# Display the unique values in the 'Work/Study Hours' column
unique_values1 = data['Dietary Habits'].unique()
print("Unique values in 'Dietary Habits':", unique_values1)


# In[35]:


#Healthy → 0,Moderate → 1,Others → 2,Unhealthy → 3
# Encode 'Dietary Habits'
data["Dietary Habits"] = label_encoder.fit_transform(data["Dietary Habits"])


# In[36]:


data.head()


# In[37]:


# Display the unique values in the 'Work/Study Hours' column
unique_values2 = data['Sleep Duration'].unique()
print("Unique values in 'Sleep Duration':", unique_values2)


# In[38]:


# Mapping categories to numeric values
sleep_mapping = {
    "'5-6 hours'": 5.5,  # Average of 5 and 6
    "'Less than 5 hours'": 4,
    "'7-8 hours'": 7.5,  # Average of 7 and 8
    "'More than 8 hours'": 9,
    "'Others'": None  # Treat as missing or unknown
}

# Apply mapping
data["Sleep Duration"] = data["Sleep Duration"].replace(sleep_mapping)


# In[39]:


data.head()


# In[40]:


# Correlation heatmap
numeric_df = data.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[41]:


# Display the unique values in the 'Work/Study Hours' column
unique_values3 = data['Degree'].unique()
print("Unique values in 'Degree':", unique_values3)


# In[42]:


# Define mapping for broader categories
degree_mapping = {
    'B.Pharm': 'UG', 'BSc': 'UG', 'BA': 'UG', 'BCA': 'UG', 'BE': 'UG', 'B.Com': 'UG', 
    'B.Ed': 'UG', 'B.Arch': 'UG', 'BHM': 'UG', 'BBA': 'UG', 'Class 12': 'Pre-UG',
    'M.Tech': 'PG', 'MSc': 'PG', 'M.Ed': 'PG', 'MA': 'PG', 'MBA': 'PG', 'M.Com': 'PG',
    'M.Pharm': 'PG', 'MCA': 'PG', 'MHM': 'PG', 'ME': 'PG', 
    'PhD': 'Doctorate', 'MD': 'Doctorate', 'LLM': 'PG', 'MBBS': 'UG', 'LLB': 'UG', 
    'Others': 'Unknown'
}

# Apply the mapping
data["Degree"] = data["Degree"].replace(degree_mapping)


# In[43]:


data.head()


# In[44]:


# Count the occurrences for each combination
financial_counts = data.groupby(["Degree", "Depression"]).size().unstack()


# Plot the bar chart
financial_counts.plot(kind="bar", figsize=(10, 6), color=["steelblue", "lightcoral"], edgecolor="black")

# Customize the plot
plt.title("Depression vs Degree", fontsize=14)
plt.xlabel("Degree", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=0)  # Keep x-axis labels horizontal
plt.legend(title="Depression", labels=["No Depression", "Depression"])
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.show()


# In[45]:



# Define the mapping for numeric encoding
degree_numeric_mapping = {
    'Pre-UG': 0,      # Pre-University
    'UG': 1,          # Undergraduate
    'PG': 2,          # Postgraduate
    'Doctorate': 3,   # Doctorate Level
    'Unknown': -1     # Unknown or Others
}

# Apply the mapping to convert to numeric values
data["Degree"] = data["Degree"].map(degree_numeric_mapping)


# In[46]:


data.head()


# In[47]:


# List of columns to convert
columns_to_convert = [
    "Sleep Duration","Degree"
]

# Step 1: Replace invalid string patterns with NaN
data[columns_to_convert] = data[columns_to_convert].apply(pd.to_numeric, errors='coerce')

# Step 2: Handle missing values (e.g., replace NaN with default values or mean)
data = data.fillna(0)  # Replace NaN with 0 (you can replace with mean or median instead)

# Step 3: Convert to integers
data[columns_to_convert] = data[columns_to_convert].astype(int)


# In[48]:


data.head()


# In[49]:


data.shape


# In[ ]:


#data.to_csv('C:\\Users\\DEVIL\\Desktop\\6th project\\DataSet\\cleandata.csv', encoding='ISO-8859-1',)


# # Model

# In[50]:


from sklearn.model_selection import train_test_split
X = data.drop("Depression", axis=1)
y = data["Depression"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[51]:


from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Preprocessing
categorical_features = ['Gender', 'Degree', 'Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
numerical_features = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Sleep Duration', 'Work/Study Hours', 'Financial Stress']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_encoded = encoder.fit_transform(X_train[categorical_features])
X_test_encoded = encoder.transform(X_test[categorical_features])

# Get feature names after one-hot encoding
feature_names_encoded = encoder.get_feature_names_out(categorical_features)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_features])
X_test_scaled = scaler.transform(X_test[numerical_features])

# Combine feature names
feature_names = np.concatenate([feature_names_encoded, numerical_features])

X_train_processed = pd.DataFrame(data=np.concatenate((X_train_encoded, X_train_scaled), axis=1), columns=feature_names)
X_test_processed = pd.DataFrame(data=np.concatenate((X_test_encoded, X_test_scaled), axis=1), columns=feature_names)


# # Random Forest

# In[52]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
# Ensure features contains the column names
features = X_train.columns


# In[53]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, ConfusionMatrixDisplay,confusion_matrix,accuracy_score

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
#Accuracy
accuracy_Ran = accuracy_score(y_test, y_pred)


# Plotting the Confusion Matrix as a Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Depression", "Depression"], 
            yticklabels=["No Depression", "Depression"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("Random Forest Accuracy:", accuracy_Ran)


# In[54]:


# ROC Curve (only if your target has binary classification)
if len(set(y_test)) == 2:
    y_pred_prob = rf_model.predict_proba(X_test)[:, 1]  # Probability scores for the positive class
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    auc_score = roc_auc_score(y_test, y_pred_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()


# In[55]:


# Get feature importances
feature_importances = rf_model.feature_importances_
features = X.columns

# Sort features by importance
sorted_idx = feature_importances.argsort()

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()


# # Logistic Regression

# In[56]:


# Logistic Regression
model_lr = LogisticRegression()
model_lr.fit(X_train_processed, y_train)
y_pred_lr = model_lr.predict(X_test_processed)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr}")


# In[57]:


# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[58]:


# Feature Importance
importances_lr = model_lr.coef_[0]
sorted_idx = importances_lr.argsort()

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), importances_lr[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.title(f"Feature Importance - Logistic Regression")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()


# # Decision Tree

# In[59]:


from sklearn.tree import DecisionTreeClassifier


# In[60]:


# Decision Tree
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_processed, y_train)
y_pred_dt = model_dt.predict(X_test_processed)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt}")


# In[61]:


# Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - Decision Tree")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[62]:


# Feature Importance
importances_dt = model_dt.feature_importances_
sorted_idx = importances_dt.argsort()

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), importances_dt[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.title(f"Feature Importance - Decision Tree")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()


# #  Support Vector Machine

# In[63]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance


# In[64]:


# Support Vector Machine
model_svm = SVC()
model_svm.fit(X_train_processed, y_train)
y_pred_svm = model_svm.predict(X_test_processed)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Support Vector Machine Accuracy: {accuracy_svm}")


# In[65]:


# Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - Support Vector Machine")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[ ]:


from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np

# Fit the SVM model
model_svm = SVC(probability=True)
model_svm.fit(X_train_processed, y_train)

# Compute permutation importance
perm_importance = permutation_importance(model_svm, X_test_processed, y_test, scoring='accuracy')

# Extract the importance scores
feature_importance = perm_importance.importances_mean

# Plot the feature importance
plt.figure(figsize=(10, 6))
indices = np.argsort(feature_importance)[::-1]
plt.bar(range(X_test_processed.shape[1]), feature_importance[indices], align="center")
plt.xticks(range(X_test_processed.shape[1]), indices)
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.title("Feature Importance - SVM (Permutation)")
plt.show()


# # K-Nearest Neighbors

# In[70]:


# K-Nearest Neighbors
model_knn = KNeighborsClassifier()
model_knn.fit(X_train_processed, y_train)
y_pred_knn = model_knn.predict(X_test_processed)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"K-Nearest Neighbors Accuracy: {accuracy_knn}")


# In[71]:


# Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - K-Nearest Neighbors")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[ ]:


# Feature Importance (using permutation importance)
result = permutation_importance(model_knn, X_train_processed, y_train, n_repeats=10, random_state=42)
importances_knn = result.importances_mean
sorted_idx = importances_knn.argsort()

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), importances_knn[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.title(f"Feature Importance - K-Nearest Neighbors")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()


# # Naive Bayes

# In[72]:


# Naive Bayes
model_nb = GaussianNB()
model_nb.fit(X_train_processed, y_train)
y_pred_nb = model_nb.predict(X_test_processed)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {accuracy_nb}")


# In[73]:


# Confusion Matrix
cm_nb = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - Naive Bayes")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[ ]:


# Feature Importance (using permutation importance)
result = permutation_importance(model_nb, X_train_processed, y_train, n_repeats=10, random_state=42)
importances_nb = result.importances_mean
sorted_idx = importances_nb.argsort()

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), importances_nb[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.title(f"Feature Importance - Naive Bayes")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()


# # Model Comprasion

# In[79]:


# Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
}

# Train and evaluate models
results = {}
for model_name, model in models.items():
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy

# Model Comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(results.keys(), results.values(), color=['blue', 'green', 'red', 'orange', 'purple', 'brown'])
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

# Add accuracy values as labels on the bars
for bar, accuracy in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{accuracy:.4f}", 
             ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()
plt.show()

# Print accuracy values
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.2f}")


# 1. Q&A
# What is the best performing model based on validation accuracy?
# The Support Vector Machine (SVM) achieved the highest accuracy of 0.8355 on the validation set, making it the best performing model among the six trained models.
# 
# What is the accuracy of each of the six models on the validation set? 
# The accuracy scores for each model are as follows:
# 
# Logistic Regression: 0.8315
# Decision Tree: 0.7573
# Random Forest: 0.8222
# Support Vector Machine: 0.8355
# K-Nearest Neighbors: 0.8047
# Naive Bayes: 0.8022
#     
# 2. Data Analysis Key Findings
# SVM Achieved Highest Validation Accuracy: The Support Vector Machine (SVM) model achieved the highest accuracy of 0.8355 on the validation set, demonstrating its superior performance compared to the other models.
# Logistic Regression & Random Forest Performed Well: Logistic Regression (0.8315) and Random Forest (0.8222) models also showed good accuracy on the validation set, indicating their potential for effective prediction.
# Decision Tree Showed Lower Accuracy: The Decision Tree model had a significantly lower accuracy of 0.7573 compared to the other models, highlighting the potential limitations of this model for the given dataset and task.
# KNN and Naive Bayes Accuracy: K-Nearest Neighbors (0.8047) and Naive Bayes (0.8022) had similar accuracy scores, showcasing their moderate performance on the validation set.
#     
# 3. Insights or Next Steps
# Further Evaluate SVM Model: Evaluate the SVM model on the test set to get a more realistic estimate of its generalization performance.
# Hyperparameter Tuning: Consider performing hyperparameter tuning for the top-performing models (e.g., SVM, Logistic Regression, Random Forest) to potentially improve their accuracy further.

# In[ ]:




