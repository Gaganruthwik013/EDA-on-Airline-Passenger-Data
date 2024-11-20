#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# Load your dataset

import chardet

# Open the file in binary mode and read a portion
with open('airlinedataset.csv', 'rb') as file:
    result = chardet.detect(file.read(10000))
    print(result)
df = pd.read_csv('airlinedataset.csv', encoding=result['encoding'])


# In[2]:


pip install pandas matplotlib seaborn plotly scikit-learn


# In[3]:


# Check for missing values
df.isnull().sum()

# Fill missing values (Example: Filling missing Age with the median value)
df['Age'].fillna(df['Age'].median(), inplace=True)

# Drop rows with missing Flight Status
df.dropna(subset=['Flight Status'], inplace=True)

# Correct data types
df['Departure Date'] = pd.to_datetime(df['Departure Date'])

# Remove duplicates
df.drop_duplicates(inplace=True)

# Check the data after cleaning
df.info()
df.head(10)


# In[4]:


df.head(10)


# In[5]:


# Step 1: Handle Missing Values

# Check for missing values
print(df.isnull().sum())

# Fill missing values for numerical columns with the median
df.fillna(df.median(numeric_only=True), inplace=True)

# Fill missing values for categorical columns with the mode
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Step 2: Convert Data Types

# Convert 'Departure Date' to datetime
df['Departure Date'] = pd.to_datetime(df['Departure Date'], errors='coerce')

# Convert categorical columns to category type
categorical_columns = ['Gender', 'Nationality', 'Airport Name', 'Country Name', 'Airport Continent', 
                       'Arrival Airport', 'Pilot Name', 'Flight Status']
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.astype('category'))

# Step 3: Remove Duplicates

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Step 4: Standardize Text

# Convert all text columns to lowercase
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.str.lower())

# Step 5: Remove Unnecessary Columns

# If there are columns that are not needed, drop them (example below assumes 'First Name' and 'Last Name' are not needed)
df.drop(['First Name', 'Last Name'], axis=1, inplace=True)

# Display cleaned dataset
print(df.head())

# Final check of the cleaned dataset
print(df.info())


# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a boxplot for Age vs. Flight Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='Flight Status', y='Age', data=df, palette='Set2')

# Set the title and labels
plt.title('Boxplot of Age vs. Flight Status')
plt.xlabel('Flight Status')
plt.ylabel('Age')

# Display the plot
plt.show()


# In[7]:


import matplotlib.pyplot as plt

# Calculate the count of each Flight Status
flight_status_counts = df['Flight Status'].value_counts()

# Plot a pie chart for the Flight Status distribution
plt.figure(figsize=(8, 6))
plt.pie(flight_status_counts, labels=flight_status_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
plt.title('Flight Status Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create the countplot
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Gender', hue='Flight Status', palette='Set1')

# Set plot labels and title
plt.title('Countplot: Gender vs. Flight Status')
plt.xlabel('Gender')
plt.ylabel('Count')

# Show the plot
plt.show()


# In[9]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Label encode the 'Gender' column
label_encoder = LabelEncoder()
df['Gender_Encoded'] = label_encoder.fit_transform(df['Gender'])

# Label encode the 'Flight Status' column
df['Flight Status Encoded'] = label_encoder.fit_transform(df['Flight Status'])

# Define the features for PCA (including the encoded 'Flight Status' and 'Gender')
features = ['Age', 'Gender_Encoded', 'Flight Status Encoded']  # Modify this if needed

# Standardize the data before applying PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

# Apply PCA
pca = PCA(n_components=2)  # Reducing to 2 components for visualization
pca_result = pca.fit_transform(scaled_data)

# Create a DataFrame for the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])

# Plot the PCA result as a scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', data=pca_df, hue=df['Flight Status'], palette='Set1', alpha=0.7)

# Add title and labels
plt.title('PCA: Dimensionality Reduction on Airline Data', fontsize=14)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend(title='Flight Status', loc='best')

# Show the plot
plt.show()


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns

# Group the data by 'Nationality' and 'Flight Status' and count the flights
flight_status_by_nationality = df.groupby(['Nationality', 'Flight Status']).size().reset_index(name='Count')

# Get the total flight counts per nationality
nationality_flight_counts = flight_status_by_nationality.groupby('Nationality')['Count'].sum().reset_index()

# Sort the nationalities by flight count in descending order and select the top 10
top_10_nationalities = nationality_flight_counts.sort_values(by='Count', ascending=False).head(10)['Nationality']

# Filter the data to include only the top 10 nationalities
top_10_data = flight_status_by_nationality[flight_status_by_nationality['Nationality'].isin(top_10_nationalities)]

# Plot the bar chart
plt.figure(figsize=(12, 8))
sns.barplot(data=top_10_data, x='Nationality', y='Count', hue='Flight Status')

# Set labels and title
plt.title('Top 10 Nationalities vs. Flight Status')
plt.xlabel('Nationality')
plt.ylabel('Count of Flights')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()

# Show the plot
plt.show()


# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'df' is your dataframe

# Step 1: Calculate the IQR for the 'Age' column
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1

# Step 2: Define the lower and upper bounds to identify outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 3: Identify the outliers
outliers = df[(df['Age'] < lower_bound) | (df['Age'] > upper_bound)]

# Step 4: Visualize the outliers using a boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(df['Age'], vert=False)
plt.title('Boxplot for Age (Detecting Outliers)')
plt.show()

# Display outliers
print("Outliers detected in Age:")
print(outliers[['Passenger ID', 'Age']])


# # 1

# In[12]:


import plotly.express as px

# Calculate gender distribution
gender_counts = df['Gender'].value_counts()

# Plot
fig = px.pie(gender_counts, 
             names=gender_counts.index, 
             values=gender_counts.values,
             title="Gender Distribution of Passengers",
             color_discrete_sequence=px.colors.sequential.Plasma)
fig.show()


# # 2

# In[13]:


plt.figure(figsize=(10, 6))
plt.hist(df['Age'], bins=30, color='#2E8B57', edgecolor='black')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# # 3

# In[14]:


plt.figure(figsize=(12, 6))
nationality_counts = df['Nationality'].value_counts().head(10)
sns.barplot(x=nationality_counts.index, y=nationality_counts.values, palette='viridis')
plt.xticks(rotation=45)
plt.title('Top 10 Most Common Nationalities of Passengers')
plt.ylabel('Number of Passengers')
plt.show()


# # 4

# In[15]:


plt.figure(figsize=(12, 8))
airport_counts = df['Airport Name'].value_counts().head(10)
sns.barplot(x=airport_counts.values, y=airport_counts.index, palette='coolwarm')
plt.title('Number of Passengers per Airport')
plt.xlabel('Number of Passengers')
plt.show()


# # 5

# In[16]:


plt.figure(figsize=(10, 6))
country_counts = df['Country Name'].value_counts().head(5)
sns.barplot(x=country_counts.index, y=country_counts.values, palette='Spectral')
plt.xticks(rotation=45)
plt.title('Top 5 Countries with Most Passenger Departures')
plt.ylabel('Number of Passengers')
plt.show()


# # 6

# In[17]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='Age', data=df, palette='Set3')
plt.title('Average Age of Passengers by Gender')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()


# # 7

# In[18]:


continent_counts = df['Airport Continent'].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(continent_counts, labels=continent_counts.index, autopct='%1.1f%%', startangle=140, pctdistance=0.85, colors=['#FFCC00', '#FF6666', '#99FF99', '#66B3FF'])
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Most Common Continent of Departure')
plt.show()


# # 8

# In[19]:


df['Departure Date'] = pd.to_datetime(df['Departure Date'])
monthly_flights = df['Departure Date'].dt.month.value_counts().sort_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x=monthly_flights.index, y=monthly_flights.values, marker='o', color='#FF4500')
plt.title('Number of Flights Recorded Per Month')
plt.xlabel('Month')
plt.ylabel('Number of Flights')
plt.xticks(range(1, 13))
plt.show()


# # 9

# In[20]:


plt.figure(figsize=(8, 6))
sns.countplot(x='Flight Status', data=df, palette='Pastel1')
plt.title('Distribution of Flights by Flight Status')
plt.xlabel('Flight Status')
plt.ylabel('Count')
plt.show()


# # 10

# In[21]:


arrival_airports = df['Arrival Airport'].value_counts().head(5)
plt.figure(figsize=(8, 6))
sns.heatmap(arrival_airports.to_frame(), annot=True, cmap='Blues', cbar=False, linewidths=0.5)
plt.title('Top 5 Arrival Airports')
plt.show()


# # 11

# In[22]:


plt.figure(figsize=(12, 6))
sns.violinplot(x='Nationality', y='Age', data=df[df['Nationality'].isin(df['Nationality'].value_counts().head(10).index)], palette='muted')
plt.xticks(rotation=45)
plt.title('Age Distribution of Passengers by Top 10 Nationalities')
plt.show()


# # 12

# In[23]:


top_airports = df['Airport Name'].value_counts().head(5).index
df_top_airports = df[df['Airport Name'].isin(top_airports)]
top_nationalities_per_airport = df_top_airports.groupby(['Airport Name', 'Nationality']).size().unstack(fill_value=0).head(5)

top_nationalities_per_airport.plot(kind='bar', stacked=True, colormap='Set1', figsize=(12, 8))
plt.title('Top 5 Nationalities of Passengers per Airport')
plt.xlabel('Airport Name')
plt.ylabel('Number of Passengers')
plt.show()


# # 13

# In[24]:


def categorize_age(age):
    if age < 18:
        return 'Child'
    elif age < 60:
        return 'Adult'
    else:
        return 'Senior'

df['Age Group'] = df['Age'].apply(categorize_age)
age_group_counts = df['Age Group'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(age_group_counts, labels=age_group_counts.index, autopct='%1.1f%%', colors=['#FFA07A', '#20B2AA', '#FF6347'])
plt.title('Age Group Distribution of Passengers')
plt.show()


# # 14

# In[25]:


plt.figure(figsize=(10, 6))
sns.countplot(x='Age Group', hue='Gender', data=df, palette='Accent')
plt.title('Gender Distribution in Different Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()


# # 15

# In[26]:


pip install squarify


# In[27]:


import squarify
import matplotlib.pyplot as plt

# Prepare data for treemap
nationality_airport_counts = df.groupby('Airport Name')['Nationality'].value_counts().unstack().fillna(0)

# Plot the treemap
plt.figure(figsize=(12, 8))
squarify.plot(sizes=nationality_airport_counts.sum().values, label=nationality_airport_counts.columns, alpha=0.8)
plt.title('Nationality Distribution by Airport')
plt.axis('off')
plt.show()


# # 16

# In[28]:


departure_counts = df['Airport Name'].value_counts().head(10)
plt.figure(figsize=(12, 8))
plt.stem(departure_counts.index, departure_counts.values, basefmt=' ', use_line_collection=True, linefmt='-.', markerfmt='o', label='Departures')
plt.xticks(rotation=45)
plt.title('Airports with the Highest Number of Departures')
plt.ylabel('Number of Departures')
plt.show()


# # 17

# In[29]:


arrival_counts = df['Arrival Airport'].value_counts().head(10)
plt.figure(figsize=(12, 8))
sns.barplot(y=arrival_counts.index, x=arrival_counts.values, palette='YlGnBu')
plt.title('Top Arrival Airports with Most Flights')
plt.xlabel('Number of Flights')
plt.show()


# # 18

# In[30]:


continent_flights = df['Airport Continent'].value_counts()
plt.figure(figsize=(10, 6))
plt.scatter(continent_flights.index, continent_flights.values, s=continent_flights.values*10, alpha=0.6, color='#9932CC')
plt.title('Flight Variation by Continents')
plt.xlabel('Continent')
plt.ylabel('Number of Flights')
plt.show()


# # 19

# In[31]:


status_by_airport = df.groupby('Airport Name')['Flight Status'].value_counts().unstack().fillna(0).head(10)
status_by_airport.plot(kind='bar', stacked=True, colormap='tab10', figsize=(12, 8))
plt.title('Common Flight Status by Airport')
plt.ylabel('Number of Flights')
plt.show()


# # 20

# In[32]:


df['Departure Month'] = df['Departure Date'].dt.month
month_airport_counts = df.groupby(['Airport Name', 'Departure Month']).size().unstack(fill_value=0).head(10)

plt.figure(figsize=(12, 8))
sns.heatmap(month_airport_counts, annot=True, cmap='YlOrRd', linewidths=0.5)
plt.title('Common Departure Month per Airport')
plt.xlabel('Departure Month')
plt.ylabel('Airport Name')
plt.show()


# # 21

# In[33]:


# Convert categorical Flight Status to numerical for correlation
df['Flight Status Num'] = df['Flight Status'].astype('category').cat.codes

plt.figure(figsize=(8, 6))
sns.heatmap(df[['Age', 'Flight Status Num']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Age and Flight Status')
plt.show()


# # 22

# In[34]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='Airport Name', y='Age', data=df[df['Airport Name'].isin(df['Airport Name'].value_counts().head(10).index)], palette='Paired')
plt.xticks(rotation=45)
plt.title('Average Age of Passengers by Departure Airport')
plt.xlabel('Airport Name')
plt.ylabel('Age')
plt.show()


# # 23

# In[35]:


plt.figure(figsize=(12, 8))
pilot_counts = df['Pilot Name'].value_counts().head(10)
sns.barplot(x=pilot_counts.index, y=pilot_counts.values, palette='cubehelix')
plt.xticks(rotation=45)
plt.title('Number of Flights by Pilot Name')
plt.ylabel('Number of Flights')
plt.show()


# # 24

# In[36]:


plt.figure(figsize=(12, 8))
continent_country_flights = df.groupby(['Airport Continent', 'Country Name']).size().unstack().fillna(0)
continent_country_flights.head(5).plot(kind='bar', stacked=True, colormap='Spectral', figsize=(12, 8))
plt.title('Number of Flights per Country by Continent')
plt.xlabel('Continent')
plt.ylabel('Number of Flights')
plt.show()


# # 25

# In[37]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='Pilot Name', y='Flight Status Num', data=df[df['Pilot Name'].isin(df['Pilot Name'].value_counts().head(10).index)], palette='Set2')
plt.xticks(rotation=45)
plt.title('Pilot Performance Based on Flight Status')
plt.xlabel('Pilot Name')
plt.ylabel('Flight Status (Numerical)')
plt.show()


# # 26

# In[38]:


plt.figure(figsize=(10, 6))
sns.violinplot(x='Airport Continent', y='Age', data=df, hue='Gender', split=True, palette='Set1')
plt.title('Distribution of Passengers per Age Group by Continent')
plt.xlabel('Airport Continent')
plt.ylabel('Age')
plt.show()


# # 27

# In[39]:


df['Year'] = df['Departure Date'].dt.year
flights_per_year = df['Year'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
sns.lineplot(x=flights_per_year.index, y=flights_per_year.values, marker='o', color='#006400')
plt.title('Trend of Flights Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Flights')
plt.show()


# # 28

# In[40]:


plt.figure(figsize=(12, 8))
status_by_nationality = df.groupby('Nationality')['Flight Status'].value_counts().unstack().fillna(0).head(10)

status_by_nationality.plot(kind='area', stacked=True, colormap='viridis', figsize=(12, 8))
plt.title('Nationalities Distribution by Flight Status')
plt.xlabel('Nationality')
plt.ylabel('Number of Flights')
plt.show()


# # 29

# In[41]:


plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Data Analysis in Airline Dataset')
plt.show()


# # 30

# In[42]:


import seaborn as sns
import matplotlib.pyplot as plt

# Reduce the dataset to top 10 nationalities for better visualization
top_nationalities = df['Nationality'].value_counts().head(10).index

plt.figure(figsize=(12, 8))
sns.stripplot(x='Nationality', y='Age', data=df[df['Nationality'].isin(top_nationalities)], 
              jitter=True, palette='tab20', size=4)

plt.xticks(rotation=45)
plt.title('Relationship Between Age and Nationality of Passengers')
plt.xlabel('Nationality')
plt.ylabel('Age')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# # 31

# In[43]:


# Convert categorical data to numerical for correlation
df_encoded = df[['Age', 'Nationality', 'Flight Status', 'Airport Name']].apply(lambda x: pd.factorize(x)[0])

# Correlation Matrix
corr_matrix = df_encoded.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix: Age, Nationality, Flight Status, and Airport')
plt.show()


# In[ ]:





# 
# # 32

# In[44]:


pivot_data = pd.pivot_table(df, values='Passenger ID', 
                            index=['Airport Continent'], 
                            columns=['Gender', 'Flight Status'], 
                            aggfunc='count', fill_value=0)

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_data, cmap='YlGnBu', annot=True, fmt='d', linewidths=0.5)
plt.title('Flight Status Distribution by Gender and Airport Continent')
plt.show()


# # 33

# In[45]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df['Gender_Encoded'] = LabelEncoder().fit_transform(df['Gender'])
df['FlightStatus_Encoded'] = LabelEncoder().fit_transform(df['Flight Status'])

# Select features for clustering
X = df[['Age', 'Gender_Encoded', 'FlightStatus_Encoded']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Gender_Encoded', hue='Cluster', palette='Set2', s=100)
plt.title('K-Means Clustering: Age and Gender with Flight Status')
plt.xlabel('Age')
plt.ylabel('Gender (Encoded)')
plt.show()


# # 34

# In[46]:


sns.pairplot(df, vars=['Age'], hue='Nationality')
plt.suptitle('Pair Plot: Age vs. Nationality and Arrival Airport', y=1.02)
plt.show()


# # 35

# In[47]:


import plotly.express as px

fig = px.scatter_3d(df, x='Airport Continent', y='Nationality', z='Flight Status', 
                    color='Flight Status', title='3D Scatter: Flight Status by Continent and Nationality')
fig.show()


# # 36

# In[48]:


df['Departure Date'] = pd.to_datetime(df['Departure Date'])
df['Month'] = df['Departure Date'].dt.month

flight_cancellations = df[df['Flight Status'] == 'Canceled'].groupby('Month').size()

flight_cancellations.plot(kind='line', marker='o', figsize=(10, 6), color='red')
plt.title('Monthly Flight Cancellations')
plt.xlabel('Month')
plt.ylabel('Number of Cancellations')
plt.show()


# # 37

# In[49]:


import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Group by pilot name to calculate the number of cancellations
pilot_cancellations = df[df['Flight Status'] == 'Canceled'].groupby('Pilot Name').size()

# Reset the index to convert the result into a DataFrame
pilot_data = pilot_cancellations.reset_index(name='Cancellations')

# Filter out pilots with zero cancellations (if any)
pilot_data = pilot_data[pilot_data['Cancellations'] > 0]

# Ensure there are enough pilots for clustering
if len(pilot_data) > 1:
    # Perform hierarchical clustering
    Z = linkage(pilot_data[['Cancellations']], method='ward')

    # Plot the dendrogram
    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=pilot_data['Pilot Name'].values, leaf_rotation=90)
    plt.title('Hierarchical Clustering on Pilot Cancellations')
    plt.xlabel('Pilot Name')
    plt.ylabel('Distance')
    plt.show()
else:
    print("Not enough data for clustering.")


# # 38

# In[50]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Keep only the top 10 most frequent 'Airport Name' values
top_airports = df['Airport Name'].value_counts().nlargest(10).index
df['Airport Name'] = df['Airport Name'].where(df['Airport Name'].isin(top_airports), 'Other')

# One-hot encode 'Airport Name' and 'Airport Continent' with fewer columns
airport_data = pd.get_dummies(df[['Airport Name', 'Airport Continent']], drop_first=True)

# Apply PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(airport_data)

plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='blue', alpha=0.5)
plt.title('PCA: Dimensionality Reduction on Selected Airport Data')
plt.show()


# # 39

# In[51]:


df['Quarter'] = df['Departure Date'].dt.to_period('Q')

flight_trends = df.groupby(['Quarter', 'Flight Status']).size().unstack().fillna(0)

flight_trends.plot(kind='line', figsize=(12, 6), marker='o')
plt.title('Flight Status Trends by Quarter')
plt.xlabel('Quarter')
plt.ylabel('Number of Flights')
plt.show()


# # 40

# In[52]:


import plotly.graph_objects as go

fig = go.Figure(data=[go.Surface(z=df.pivot_table(index='Age', columns='Airport Name', 
                                                  values='Flight Status', aggfunc='count', fill_value=0).values)])

fig.update_layout(title='3D Surface Plot: Passenger Age vs. Airport Name and Flight Status')
fig.show()


# # 41

# In[53]:


plt.figure(figsize=(10, 6))
sns.violinplot(x='Gender', y='Age', hue='Flight Status', data=df, dodge=True)
plt.title('Violin Plot: Age Distribution by Gender and Flight Status')
plt.show()


# # 42

# In[54]:


import plotly.express as px

# Create a new DataFrame that counts the number of passengers by nationality and continent
passenger_counts = df.groupby(['Airport Continent', 'Nationality']).size().reset_index(name='Passenger Count')

# Create the treemap using the counts
fig = px.treemap(passenger_counts, path=['Airport Continent', 'Nationality'], values='Passenger Count', 
                 title='Treemap: Passengers by Nationality and Continent')
fig.show()


# # 43

# In[55]:


import plotly.express as px

# Limit to the top 5 most frequent airports and top 10 pilots to reduce complexity
top_airports = df['Airport Name'].value_counts().nlargest(5).index
top_pilots = df['Pilot Name'].value_counts().nlargest(10).index

filtered_df = df[df['Airport Name'].isin(top_airports) & df['Pilot Name'].isin(top_pilots)]

# Group and create the plot
flight_counts = filtered_df.groupby(['Airport Name', 'Pilot Name', 'Flight Status']).size().reset_index(name='Flight Count')
fig = px.sunburst(flight_counts, path=['Airport Name', 'Pilot Name', 'Flight Status'], 
                  values='Flight Count', title='Sunburst Plot: Flight Status by Airport and Pilot')
fig.show()



# # 44

# In[56]:


df['Age Group'] = pd.cut(df['Age'], bins=[0, 17, 40, 60, 100], labels=['Child', 'Adult', 'Middle-Aged', 'Senior'])

df['Age Group'].value_counts().plot(kind='bar', color='teal', figsize=(8, 5))
plt.title('Histogram: Distribution of Flights by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Passengers')
plt.show()


# # 45

# In[57]:


from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import LabelEncoder

# Encode categorical columns
df_encoded = df[['Age', 'Flight Status', 'Airport Continent']].copy()
df_encoded['Flight Status'] = LabelEncoder().fit_transform(df['Flight Status'])
df_encoded['Airport Continent'] = LabelEncoder().fit_transform(df['Airport Continent'])

# Plot the parallel coordinates plot
plt.figure(figsize=(12, 6))
parallel_coordinates(df_encoded, class_column='Flight Status', color=['r', 'g', 'b'])
plt.title('Parallel Coordinates Plot: Age, Flight Status, and Continent')
plt.show()


# # 46

# In[58]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='Nationality', y='Age', data=df)
plt.xticks(rotation=90)
plt.title('Box Plot: Age Distribution by Nationality')
plt.show()


# # 47

# In[59]:


df['Month'] = df['Departure Date'].dt.month

monthly_flights = df.groupby(['Month', 'Airport Continent']).size().unstack().fillna(0)

monthly_flights.plot(kind='line', marker='o', figsize=(12, 6))
plt.title('Flights by Month and Continent')
plt.xlabel('Month')
plt.ylabel('Number of Flights')
plt.show()


# # 48 

# In[60]:


plt.figure(figsize=(12, 6))
sns.countplot(x='Country Name', hue='Flight Status', data=df)
plt.xticks(rotation=90)
plt.title('Count Plot: Flight Status by Country')
plt.show()


# 
# # 49

# In[61]:


pilot_flights = df.groupby('Pilot Name').size().reset_index(name='Number of Flights')
pilot_flights['Age'] = df.groupby('Pilot Name')['Age'].mean().values

sns.regplot(x='Age', y='Number of Flights', data=pilot_flights)
plt.title('Scatter Plot: Age vs. Number of Flights per Pilot')
plt.show()


# # 50

# In[62]:


# Ensure Passenger ID is treated as numeric
df['Passenger ID'] = pd.to_numeric(df['Passenger ID'], errors='coerce')

# Drop rows where Passenger ID could not be converted to numeric
df_clean = df.dropna(subset=['Passenger ID'])

# Create the bubble chart with corrected data
import plotly.express as px

fig = px.scatter(
    df_clean, 
    x='Gender', 
    y='Nationality', 
    size='Passenger ID', 
    color='Flight Status', 
    title='Bubble Chart: Flight Status by Gender and Nationality',
    size_max=60  # Adjust size for better visuals
)
fig.show()


# # 51

# In[63]:


df['Day of Week'] = df['Departure Date'].dt.day_name()

heatmap_data = pd.crosstab(df['Day of Week'], df['Flight Status'])

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='d')
plt.title('Heatmap: Flights by Day of Week and Flight Status')
plt.show()


# # 52 

# In[64]:


top_arrival_airports = df['Arrival Airport'].value_counts().nlargest(5)

top_arrival_airports.plot(kind='bar', color='purple', figsize=(8, 5))
plt.title('Top 5 Arrival Airports with Most Flights')
plt.show()


# # 53

# In[65]:


df_corr = pd.get_dummies(df[['Flight Status', 'Gender', 'Age', 'Country Name']], drop_first=True).corr()

plt.figure(figsize=(10, 6))
sns.heatmap(df_corr, annot=True, cmap='viridis', linewidths=0.5)
plt.title('Correlation Heatmap of Flight Status vs. Other Variables')
plt.show()


# # 54

# In[66]:


import pandas as pd
import plotly.express as px

# Ensure 'Departure Date' is in datetime format
df['Departure Date'] = pd.to_datetime(df['Departure Date'], errors='coerce')

# Drop rows with missing or invalid dates
df = df.dropna(subset=['Departure Date'])

# Convert 'Departure Date' to date only (no time component)
df['Departure Date'] = df['Departure Date'].dt.date

# Create the animated scatter plot
fig = px.scatter(
    df, 
    x='Age', 
    y='Passenger ID', 
    animation_frame='Departure Date', 
    color='Airport Continent', 
    title='Animated Scatter: Passengers over Time'
)

fig.show()


# # 55

# In[67]:


plt.figure(figsize=(8, 5))
sns.boxenplot(x='Flight Status', y='Age', data=df)
plt.title('Boxen Plot: Age vs. Flight Status')
plt.show()


# # 56

# In[68]:


fig = px.scatter_3d(df, x='Age', y='Gender', z='Flight Status', color='Flight Status',
                    title='3D Scatter Plot: Age, Gender, and Flight Status')
fig.show()


# # 57

# In[69]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Filter delayed flights and group by Pilot Name to get the top 10 pilots with most delays
delayed_flights = df[df['Flight Status'] == 'Delayed'].groupby('Pilot Name').size().nlargest(10)

# Ensure the data is in the correct format for a heatmap
if delayed_flights.empty:
    print("No delayed flights found.")
else:
    plt.figure(figsize=(8, 5))
    # Convert the Series to a DataFrame with proper labels for the heatmap
    delayed_flights_df = delayed_flights.to_frame(name='Number of Delays')
    
    # Create the heatmap
    sns.heatmap(delayed_flights_df, annot=True, cmap='Blues', cbar=False, linewidths=0.5)
    plt.title('Top 10 Pilots with Most Delays')
    plt.ylabel('Pilot Name')
    plt.xlabel('Number of Delays')
    plt.show()


# # 58

# In[70]:


get_ipython().system('pip install joypy')


# In[71]:


import joypy
import matplotlib.pyplot as plt
from matplotlib import cm  # Import Matplotlib's colormap

# Create a joyplot by grouping data by Gender and plotting Age distributions
plt.figure(figsize=(10, 6))
joypy.joyplot(
    df, 
    by='Gender', 
    column='Age', 
    colormap=cm.coolwarm  # Use a colormap object instead of a string
)

plt.title('Joyplot: Age Distribution by Gender')
plt.show()


# # 

# # 59

# In[72]:


flight_status_by_continent = df.groupby(['Airport Continent', 'Flight Status']).size().unstack().fillna(0)

flight_status_by_continent.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Stacked Bar Plot: Flight Status by Continent')
plt.show()


# # 60

# In[73]:


gender_distribution = df['Gender'].value_counts()

gender_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=140, figsize=(8, 8), colors=['#ff9999','#66b3ff'])
plt.title('Pie Chart: Gender Distribution of Passengers')
plt.ylabel('')
plt.show()


# In[ ]:





# # 61

# In[74]:


#average age of passengers by gender?
df.groupby('Gender')['Age'].mean()


# 
# # 62

# In[75]:


# Count of passengers by nationality
df['Nationality'].value_counts()


# # 63

# In[76]:


# Age distribution of passengers by continent
df.groupby('Airport Continent')['Age'].describe()


# # 64

# In[77]:


# Count of flight statuses for each airport
df.groupby('Airport Name')['Flight Status'].value_counts()


# # 65

# In[78]:


# Most common flight status overall
df['Flight Status'].value_counts().idxmax()


# # 66

# In[79]:


# List of unique departure dates
df['Departure Date'].unique()


# # 67

# In[80]:


# Average age of passengers for each flight status
df.groupby('Flight Status')['Age'].mean()


# # 68

# In[81]:


# Count of male and female passengers by continent
df.groupby(['Airport Continent', 'Gender']).size()


# # 69

# In[82]:


# Top 5 most common nationalities
df['Nationality'].value_counts().head(5)


# # 70

# In[83]:


# Percentage of each flight status
df['Flight Status'].value_counts(normalize=True) * 100


# # 71

# In[84]:


# Count of passengers by age group
df['Age Group'].value_counts()


# # 72

# In[85]:


# Top 3 countries with most flights
df['Country Name'].value_counts().head(3)


# # 73
# 

# In[86]:


# Average age by arrival airport
df.groupby('Arrival Airport')['Age'].mean()


# # 74

# In[87]:


# Flight count by departure month
df['Departure Month'].value_counts()


# # 75

# In[88]:


# Count of flights by gender and flight status
df.groupby(['Gender', 'Flight Status']).size()


# # 76

# In[89]:


# Average age by departure month
df.groupby('Departure Month')['Age'].mean()


# # 77

# In[90]:


# Flight count by day of week
df['Day of Week'].value_counts()


# # 78

# In[91]:


# Flights per airport continent
df['Airport Continent'].value_counts()


# # 79

# In[92]:


# Passenger age distribution by gender
df.groupby('Gender')['Age'].describe()


# # 80

# In[93]:


# Average age of passengers per quarter
df.groupby('Quarter')['Age'].mean()


# # 81

# In[94]:


# Top 5 most frequent pilots
df['Pilot Name'].value_counts().head(5)


# # 83

# In[95]:


# Total flights by year
df['Year'].value_counts()


# # 84

# In[96]:


# Count of passengers by cluster
df['Cluster'].value_counts()


# # 85

# In[97]:


# Count of flights by month
df['Month'].value_counts()


# # 86

# In[98]:


# Average age by gender and nationality
df.groupby(['Gender', 'Nationality'])['Age'].mean()


# # 87

# In[99]:


# Flight count by gender and year
df.groupby(['Gender', 'Year']).size()


# # 88

# In[100]:


# Top 3 busiest airports in each country
df.groupby('Country Name')['Airport Name'].value_counts().groupby(level=0).head(3)


# # 89

# In[101]:


# Average age of passengers per continent
df.groupby('Airport Continent')['Age'].mean()


# # 90

# In[102]:


# Flight status distribution per age group
df.groupby('Age Group')['Flight Status'].value_counts()


# # 91

# In[103]:


# Top 3 most common departure months
df['Departure Month'].value_counts().head(3)


# # 92

# In[104]:


# Average age of passengers by cluster
df.groupby('Cluster')['Age'].mean()


# # 93

# In[105]:


# Flight status count per departure month
df.groupby('Departure Month')['Flight Status'].value_counts()


# # 94

# In[106]:


# Count of flights by pilot name and gender
df.groupby(['Pilot Name', 'Gender']).size()


# # 95

# In[107]:


# Flight count by country name and flight status
df.groupby(['Country Name', 'Flight Status']).size()


# # 96

# In[108]:


# Top 5 airports with most canceled flights
df[df['Flight Status'] == 'Canceled']['Airport Name'].value_counts().head(5)


# # 97

# In[109]:


# Average age of passengers per year
df.groupby('Year')['Age'].mean()


# # 98

# In[110]:


# Flight status count per airport continent
df.groupby('Airport Continent')['Flight Status'].value_counts()


# # 99

# In[111]:


# Total number of passengers per airport
df['Airport Name'].value_counts()


# # 100

# In[113]:


# Most common age group in each continent
df.groupby('Airport Continent')['Age Group'].agg(lambda x: x.value_counts().index[0])


# In[ ]:




