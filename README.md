# Build a Collaborative Filtering Recommender System to suggest movie to each use

## Procedure
* Step 1: read file .base => use pandas
  code example colum_name = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
               ratings_base = pd.read_csv('ub.base', sep='\t', names=colum_name, encoding='latin-1')
* Step 2: build a normalized utility matrix => use numpy. 
* Step 3: normalize matrix
* Step 3: build Cosine Similarity to calculate relation between each user
* Step 4: calculate rating prediction
             Item - Item base is similar with User - User base but you tranpose normalized utility matrix
          all step are similar 
