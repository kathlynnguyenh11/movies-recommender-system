# Assignment : Build a Collaborative Filtering Recommender System to suggest movie to each use
## Instruction:
<em> I have attached the model file under .base file and colum name includes (user_id, movie_id, rating, unix_timestamp) </br>
Mission : Build a Collaborative Filtering Recommender System using Python and similarity argorithms
I want you build it in 2 type user-user base and item-item base. I suggest you use library numpy pandas to implement it. </em>

## Guide line : User - User base
* step 1: read file .base => use pandas
  code example colum_name = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
               ratings_base = pd.read_csv('ub.base', sep='\t', names=colum_name, encoding='latin-1')
* step 2: build a normalized utility matrix => use numpy. 
* step 3: normalize matrix
* step 3: build Cosine Similarity to calculate relation between each user
* step 4: calculate rating prediction
             Item - Item base is similar with User - User base but you should tranpose normalized utility matrix
          all step are similar 