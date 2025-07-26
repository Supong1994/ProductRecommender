# ProductRecommender
This project is a user-based collaborative filtering recommendation system designed to suggest shopping products to users based on their historical preferences and product ratings. It mimics real-world e-commerce platforms like Amazon or Flipkart, where recommendations are made using similarity in user behavior.
The system uses a custom dataset that includes user shopping histories such as purchased product categories, prices, and user-given ratings.
How It Works
The system uses the User-Based Collaborative Filtering approach with the following steps:

Data Preprocessing:

Input data is a CSV file containing user interactions with products: user_id, product_id, product_name, category, price, and rating.

Product names are cleaned for consistency (e.g., trimming spaces, converting to lowercase).

User-Item Matrix Construction:

A pivot table (user-item matrix) is built where rows represent users and columns represent products, with values being user ratings.

Unrated products are marked as 0.

User Similarity Computation:

The NearestNeighbors algorithm from scikit-learn is used to compute the cosine similarity between users.

For a given user, the most similar users are identified based on rating patterns.

Recommendation Generation:

Products rated by similar users but not yet rated by the target user are collected.

For each such product, the system computes an average rating from similar users.

Top N (e.g., 3) products are recommended based on the highest average ratings.

📌 Features
Simple, clean implementation of user-based collaborative filtering

Dynamic user similarity calculation using cosine distance

Easy to extend with new users, products, and ratings

Clear visualization and output

🧰 Possible Extensions
Add item-based collaborative filtering

Include content-based filtering using product metadata

Integrate with a web interface (e.g., Flask or Streamlit)

Use real-world datasets like Amazon Reviews or MovieLens
