#!/usr/bin/env python
# coding: utf-8

# In[1]:


NAME = "Somanich Bunlee"
COLLABORATORS = ""


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import ast  # For parsing JSON-like strings


# In[3]:


# Load the dataset
movies_df = pd.read_csv(r'../data/tmdb_5000_movies.csv')

# Quick Overview
print(movies_df.info())
print(movies_df.head())


# ### Data Cleaning and Preprocessing

# In[4]:


# Drop duplicate rows if any
movies_df = movies_df.drop_duplicates()

# Handle missing values
movies_df = movies_df.dropna(subset=['budget', 'revenue', 'genres', 'popularity'])

# Convert budget and revenue to numeric (ensure they're in numeric format)
movies_df['budget'] = pd.to_numeric(movies_df['budget'], errors='coerce')
movies_df['revenue'] = pd.to_numeric(movies_df['revenue'], errors='coerce')

# Fill or drop missing data in critical columns (e.g., runtime)
movies_df['runtime'] = movies_df['runtime'].fillna(movies_df['runtime'].median())


# ### Feature Engineering
# 

# In[5]:


# Create profit_margin
movies_df['profit_margin'] = (movies_df['revenue'] - movies_df['budget']) / movies_df['budget']

# Extract genres into a list
import ast
movies_df['genres_list'] = movies_df['genres'].apply(lambda x: [genre['name'] for genre in ast.literal_eval(x)] if pd.notnull(x) else [])


# ### Exploratory Data Analysis (EDA)
# Relationship Between Genres, Popularity, and Revenue

# In[6]:


# Relationship Between Genres, Popularity, and Revenue

# Exploding genres for analysis
movies_exploded = movies_df.explode('genres_list')

# Average popularity and revenue by genre
genre_analysis = movies_exploded.groupby('genres_list')[['popularity', 'revenue', 'vote_average']].mean().sort_values(by='popularity', ascending=False)
print(genre_analysis)

# Visualizing Average Popularity, Revenue, and Rating by Genre 
plt.figure(figsize=(12, 6))
sns.barplot(data=genre_analysis.reset_index(), x='genres_list', y='popularity', palette='rocket')
plt.title('Average Popularity by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Popularity')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=genre_analysis.reset_index(), x='genres_list', y='revenue', palette='rocket')
plt.title('Average Revenue by Genre')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=genre_analysis.reset_index(), x='genres_list', y='vote_average', palette='rocket')
plt.title('Average Rating by Genre')
plt.xticks(rotation=45)
plt.show()


# Release Year and Runtime Analysis

# In[7]:


# Converting release_date to datetime and extracting year
movies_df['release_year'] = pd.to_datetime(movies_df['release_date'], errors='coerce').dt.year

# Average popularity and revenue by release year
year_analysis = movies_df.groupby('release_year')[['popularity', 'revenue']].mean()
print(year_analysis)

# Visualizing Popularity and Revenue trends over the years 
plt.figure(figsize=(12, 6))
sns.lineplot(data=year_analysis, x=year_analysis.index, y='popularity', color='purple')
plt.title('Popularity Trends Over the Years')
plt.xlabel('Release Year')
plt.ylabel('Average Popularity')
plt.show()

# Correlation heatmap including profit_margin
movies_df['profit_margin'] = (movies_df['revenue'] - movies_df['budget']) / movies_df['budget']
correlation_matrix = movies_df[['budget', 'revenue', 'popularity', 'runtime', 'profit_margin']].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='rocket')
plt.title('Correlation Heatmap')
plt.show()


# In[8]:


# Pair plots of essential columns (Not included in the paper)
sns.pairplot(movies_df[['budget', 'revenue', 'popularity', 'runtime']], diag_kind='kde')
plt.show()


# ### Clustering for Recommendations

# In[9]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# Select Relevant Features for clustering
clustering_features = movies_df[['popularity', 'revenue', 'budget']].fillna(0)

# Scale standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(clustering_features)

# Determine the optimal number of clusters (Elbow Method or Silhouette Score)
range_n_clusters = list(range(2, 11))
silhouette_scores = []

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_features)
    silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))

# Plot silhouette scores
plt.plot(range_n_clusters, silhouette_scores, marker='o', color='purple')
plt.title("Silhouette Scores for Different Cluster Numbers")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
movies_df['cluster'] = kmeans.fit_predict(scaled_features)

# Visualize clusters
sns.scatterplot(data=movies_df, x='popularity', y='revenue', hue='cluster', palette='PuRd')
plt.title('Clusters of Movies by Popularity and Revenue')
plt.show()


# In[10]:


# View Cluster Insights with K = 2
cluster_insights = movies_df.groupby('cluster')[['budget', 'runtime', 'popularity', 'vote_average']].mean()
print(cluster_insights)


# ### Content-Based Recommendation System

# In[11]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Assuming `movies_df` contains columns: 'title', 'genres_list', 'overview', etc.

# Function to recommend movies based on genre
def recommend_movies_by_genre(input_genre, top_n=5):
    # Filter movies by the genre
    filtered_movies = movies_df[movies_df['genres_list'].apply(lambda x: input_genre in x)]
    # Sort by popularity and return the top_n movies
    recommended = filtered_movies.sort_values('popularity', ascending=False).head(top_n)
    return recommended[['title', 'vote_average', 'genres_list', 'overview']]

# Function to recommend movies based on the movie title (content-based filtering)
def recommend_movies_by_title(input_title, top_n=5):
    # Ensure the movie exists in the dataset
    if input_title not in movies_df['title'].values:
        return "Movie not found in the dataset!"
    
    # Create a TF-IDF vectorizer to convert the 'overview' or other text-based features into vectors
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit the vectorizer on the 'overview' column or other features
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['overview'].fillna(''))
    
    # Compute the cosine similarity between the input movie and all other movies
    input_movie_index = movies_df[movies_df['title'] == input_title].index[0]
    cosine_similarities = cosine_similarity(tfidf_matrix[input_movie_index], tfidf_matrix).flatten()
    
    # Get the indices of the most similar movies (excluding the input movie itself)
    similar_indices = cosine_similarities.argsort()[-top_n-1:-1][::-1]
    
    # Return the top_n similar movies
    similar_movies = movies_df.iloc[similar_indices]
    return similar_movies[['title', 'vote_average', 'genres_list', 'overview']]

# Main function to take input from the user
def recommend_movies(input_value, top_n=5):
    # Check if the input is a movie title or a genre
    if input_value in movies_df['title'].values:
        # Call the title-based recommendation function
        print(f"\nRecommendations based on movie title: '{input_value}'")
        recommendations = recommend_movies_by_title(input_value, top_n)
    else:
        # Call the genre-based recommendation function
        print(f"\nRecommendations based on genre: '{input_value}'")
        recommendations = recommend_movies_by_genre(input_value, top_n)
    
    print(recommendations)

# Continuous loop for getting user input
def interactive_recommendation_system():
    while True:
        # Get input from the user
        user_input = input("\nEnter a movie title or genre to get recommendations (or type 'exit' to quit): ").strip()
        
        if user_input.lower() == 'exit':
            print("Exiting the recommendation system. Goodbye!")
            break  # Exit the loop if user types 'exit'
        
        # Get top 5 recommendations
        recommend_movies(user_input, top_n=5)

# Run the interactive recommendation system
interactive_recommendation_system()


# ### Regression Analysis
# Predict popularity or revenue using features like budget, genres, and runtime.

# In[12]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Prepare features and target for regression model
features = ['budget', 'revenue', 'runtime']
target = 'popularity'

# Drop rows with missing values in features
movies_df = movies_df.dropna(subset=features)

# Split data into training and testing sets
X = movies_df[features]
y = movies_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Make predictions
y_pred = reg_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = reg_model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print('RMSE:', mean_squared_error(y_test, y_pred, squared=False))
print(f"R^2 Score: {r2:.2f}")


# In[ ]:




