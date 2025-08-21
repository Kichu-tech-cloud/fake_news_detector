import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load the datasets
fake_news = pd.read_csv("data/Fake.csv", encoding="ISO-8859-1", low_memory=False)
true_news = pd.read_csv("data/True.csv", encoding="ISO-8859-1", low_memory=False)

# Prepare the data
fake_news["label"] = 0  # Label fake news as 0
true_news["label"] = 1  # Label true news as 1

# Combine the datasets
news_data = pd.concat([fake_news, true_news], axis=0)

# Shuffle the dataset
news_data = news_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Select features and target
X = news_data["text"]  # Assuming the text column contains the news content
y = news_data["label"]

# Handle missing data (if any)
X.fillna("", inplace=True)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate the model (Optional: You can print the accuracy here)
accuracy = model.score(X_test_tfidf, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model and vectorizer
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)


# Function to predict news type
def predict_news(news_input):
    # Load the model and vectorizer
    with open("model.pkl", "rb") as model_file:
        loaded_model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)

    # Transform the input news
    input_tfidf = loaded_vectorizer.transform([news_input])

    # Make a prediction
    prediction = loaded_model.predict(input_tfidf)

    # Return the result
    return "Real News" if prediction[0] == 1 else "Fake News"


# Accept user input and predict
if __name__ == "__main__":
    news_input = input("Enter a news article: ")
    result = predict_news(news_input)
    print(f"The given news article is: {result}")
