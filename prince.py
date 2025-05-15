import pandas as pd
from flask import Flask, request, jsonify
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

app = Flask(__name__)

# Sample user data
users_data = {
    "user_id": [1, 2, 3],
    "age": [23, 30, 26],
    "gender": ["M", "F", "F"],
    "location": ["New York", "London", "Tokyo"]
}
users_df = pd.DataFrame(users_data)

# Sample interaction data
interactions_data = {
    "user_id": [1, 1, 2, 2, 3],
    "item_id": [101, 102, 103, 104, 105],
    "interaction_score": [5, 3, 4, 2, 5]
}
interaction_df = pd.DataFrame(interactions_data)

# Load data into surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(interaction_df[['user_id', 'item_id', 'interaction_score']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

# Train recommendation model
model = SVD()
model.fit(trainset)

# Recommendation function
def get_recommendations(user_id, all_items, model):
    interacted_items = set(interaction_df[interaction_df['user_id'] == user_id]['item_id'].values)
    items_to_predict = [iid for iid in all_items if iid not in interacted_items]
    predictions = [model.predict(user_id, iid) for iid in items_to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_5 = [{"item_id": int(pred.iid), "predicted_score": round(pred.est, 2)} for pred in predictions[:5]]
    return top_5

# API Endpoint
@app.route("/recommend", methods=["GET"])
def recommend():
    try:
        user_id = int(request.args.get("user_id"))
        if user_id not in users_df["user_id"].values:
            return jsonify({"error": "User not found"}), 404

        all_items = interaction_df['item_id'].unique()
        recommendations = get_recommendations(user_id, all_items, model)

        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run app
if __name__ == "__main__":
    app.run(debug=True)
