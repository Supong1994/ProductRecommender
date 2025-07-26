import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ----------------------------------------
# Step 1: Load Dataset
# ----------------------------------------
df = pd.read_csv("D:\\Datasets\\product\\shopping_recommendation_dataset_200_users.csv")
print("Preview of dataset:\n", df.head())

# ----------------------------------------
# Step 2: Create User-Item Matrix
# ----------------------------------------
user_item_matrix = df.pivot_table(index='user_id', columns='product_name', values='rating')
user_item_matrix.fillna(0, inplace=True)

# Save product mapping
product_names = user_item_matrix.columns.tolist()
user_ids = user_item_matrix.index.tolist()

# ----------------------------------------
# Step 3: Convert to Sparse Matrix & Train Model
# ----------------------------------------
sparse_matrix = csr_matrix(user_item_matrix.values)

model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(sparse_matrix)

# ----------------------------------------
# Step 4: Recommend Products to a User (Improved Logic)
# ----------------------------------------
def recommend_to_user(target_user_id, n_neighbors=3, n_recommendations=5):
    if target_user_id not in user_ids:
        print(f"User ID {target_user_id} not found.")
        return

    user_index = user_ids.index(target_user_id)

    # Find nearest neighbors
    distances, indices = model.kneighbors([user_item_matrix.values[user_index]], n_neighbors=n_neighbors + 1)

    # Skip the first (it's the user themself)
    similar_users = indices.flatten()[1:]

    # Products already rated by the target user
    user_rated = set(user_item_matrix.loc[target_user_id][user_item_matrix.loc[target_user_id] > 0].index)

    # Gather ratings from similar users
    recs = {}
    for sim_user_idx in similar_users:
        sim_user_id = user_ids[sim_user_idx]
        sim_ratings = user_item_matrix.loc[sim_user_id]

        for product, rating in sim_ratings.items():
            if product not in user_rated and rating > 0:
                if product not in recs:
                    recs[product] = []
                recs[product].append(rating)

    # If no recommendations found
    if not recs:
        print(f"No recommendations found for User {target_user_id}.")
        return

    # Average the ratings
    averaged = {product: np.mean(ratings) for product, ratings in recs.items()}

    # Sort by highest average rating
    sorted_recs = sorted(averaged.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop {n_recommendations} product recommendations for User {target_user_id}:")
    for i, (product, score) in enumerate(sorted_recs[:n_recommendations], start=1):
        print(f"{i}. {product} (Avg Rating: {score:.2f})")
recommend_to_user(target_user_id=1002, n_neighbors=3, n_recommendations=3)