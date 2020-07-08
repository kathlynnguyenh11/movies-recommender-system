import pandas as pd 
import numpy as np 
import sklearn.metrics.pairwise as pw
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances
from IPython.display import display
import os

#INFO:
movie_title = 'Boomerang (1992)'
user_name = "Smith"

BASE_DIR = os.path.dirname(__file__)
RECOMMENDATION_PATH = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
RECOMMENDATION_PATH = os.path.join('asset','result')

def item_based(input_dataframe, title):
    pivot_item_based = pd.pivot_table(input_dataframe, index='title', columns=['user_id'],values='rating')

    sparse_pivot = sparse.csr_matrix(pivot_item_based.fillna(0))
    recommender = pw.cosine_similarity(sparse_pivot)

    recommender_df = pd.DataFrame(recommender, columns=pivot_item_based.index,index=pivot_item_based.index)

    ## Item rating based cosine similarity
    cosine_df = pd.DataFrame(recommender_df[title].sort_values(ascending=False))
    cosine_df.reset_index(level=0, inplace=True)
    cosine_df.columns = ['title', 'cosine_sim']

    return cosine_df

def user_based_recom(input_dataframe,input_user_name):    
    pivot_user_based = pd.pivot_table(input_dataframe, index='name', columns=['title'], values='rating')
    user_sparse_pivot = sparse.csr_matrix(pivot_user_based.fillna(0))
    user_recommender = pw.cosine_similarity(user_sparse_pivot)
    user_recommender_df = pd.DataFrame(user_recommender, columns=pivot_user_based.index.values,index = pivot_user_based.index.values)
    ## Movie User based Cosine Similarity data frame 
    usr_cosine_df = pd.DataFrame(user_recommender_df[input_user_name].sort_values(ascending=False))
    usr_cosine_df.reset_index(level=0, inplace=True)
    usr_cosine_df.columns = ['title','cosine_sim']
    ## 4 most similar users
    similar_usr = list(usr_cosine_df['title'][1:5].values)
    ## Comparing reviews with similar users
    similar_usr_df = pivot_user_based.T[[input_user_name] + similar_usr].fillna(0)
    similar_usr_df['Mean Score'] = similar_usr_df[similar_usr].mean(numeric_only=True,axis=1)
    similar_usr_df.sort_values('Mean Score', ascending=False,inplace = True)
    #Check user rated movies vs similar users ratings
    #display(similar_usr_df[similar_usr_df[user_id]!=0])
    return similar_usr_df[similar_usr_df[input_user_name]==0]

def generate_recomendations(df,title,user_name,top_results):
    print("Movie Recommender")
    print("Top", str(top_results), "Films you might enjooy based that you watched", str(title))
    ## Item Rating Based Cosine Similarity
    cos_sim = item_based(df,title)
    print(RECOMMENDATION_PATH)
    if not os.path.exists(RECOMMENDATION_PATH):
        os.mkdir(RECOMMENDATION_PATH)

    
    item_based_path = os.path.join(RECOMMENDATION_PATH, 'item_based.csv')
    print(item_based_path)
    
    cos_sim[1:top_results+1].to_csv(item_based_path)
    display(cos_sim[1:top_results+1])

    print("************************************************\n")
    print("Flims reccomended for you:")

    a = user_based_recom(df,user_name)
    user_based_path = os.path.join(RECOMMENDATION_PATH, 'user_based.csv')
    a[0:top_results].to_csv(user_based_path)
    display(user_based_recom(df,user_name)[0:top_results])

def __main__():
    #Read in ratings base
    column_name = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings_base = pd.read_csv('./asset/data/ub.base', sep='\t', names=column_name, encoding='latin-1')
    
    #Read in movie title base   
    title_column_name = ['movie_id', 'title','genres']
    titles_base = pd.read_csv('./asset/data/titles.csv', sep=',', names=title_column_name, encoding='latin-1')
    
    #Generate movie title hash map to map with ratings data:
    titles_dict = {}
    for index, row in titles_base.iterrows():
        titles_dict[row['movie_id']] = row['title']

    #Map movie title with movie id
    merged_df = ratings_base
    merged_df['title'] = merged_df['movie_id'].map(titles_dict)
    #merged_df.to_csv('ratings_titles.csv')

    #Read in user name base   
    user_column_name = ['user_id', 'name']
    names_base = pd.read_csv('./asset/data/names.csv', sep=',', names=user_column_name, encoding='latin-1')
    
    #Generate user names hash map to map with data:
    names_dict = {}
    for index, row in names_base.iterrows():
        names_dict[row['user_id']] = row['name']
    
    #Map user name with user id
    merged_df['name'] = merged_df['user_id'].map(names_dict)
    merged_df.to_csv('final_df.csv')

    generate_recomendations(merged_df,movie_title,user_name,5)

__main__()

