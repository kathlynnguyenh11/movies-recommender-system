import pandas as pd 
import numpy as np 
import sklearn.metrics.pairwise as pw
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances
from IPython.display import display

def item_based(input_dataframe, input_movie_id):
    pivot_item_based = pd.pivot_table(input_dataframe, index='movie_id', columns=['user_id'],values='rating')

    sparse_pivot = sparse.csr_matrix(pivot_item_based.fillna(0))
    recommender = pw.cosine_similarity(sparse_pivot)

    recommender_df = pd.DataFrame(recommender, columns=pivot_item_based.index,index=pivot_item_based.index)

    ## Item rating based cosine similarity
    cosine_df = pd.DataFrame(recommender_df[input_movie_id].sort_values(ascending=False))
    cosine_df.reset_index(level=0, inplace=True)
    cosine_df.columns = ['movie_id', 'cosine_sim']

    return cosine_df

def generate_recomendations(df,movie_id,top_results):
    print("Movie Recommender")
    print("Top", str(top_results), "Films you might enjooy based that you watched", str(movie_id))
    ## Item Rating Based Cosine Similarity
    cos_sim = item_based(df,movie_id)
    display(cos_sim[1:top_results+1])
    print("************************************************\n")

def __main__():
    movie_id = 13
    column_name = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings_base = pd.read_csv('./asset/data/ub.base', sep='\t', names=column_name, encoding='latin-1')
    generate_recomendations(ratings_base,movie_id,5)

__main__()

