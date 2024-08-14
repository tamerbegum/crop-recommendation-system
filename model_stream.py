from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib

# Load the data

df = pd.read_csv("Crop_recommendation.csv")

# Prepare the data

X = df.drop("label", axis=1)
y = df["label"]
numeric_features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Define the preprocessor

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
    ],
    remainder="passthrough"
)

# Define the pipeline with the tuned Random Forest Classifier

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=50, max_features=None, random_state=12345))
])

# Train the model
pipeline.fit(X, y)

# Save the trained pipeline to a file

joblib.dump(pipeline, "stream.joblib")

print("Model training complete and saved as 'stream.joblib'.")


# Function to get user input and make a prediction

def get_prediction():
    # Load the trained model
    pipeline = joblib.load("stream.joblib")

    # Prompt the user for input values
    N = float(input("Enter N: "))
    P = float(input("Enter P: "))
    K = float(input("Enter K: "))
    temperature = float(input("Enter Temperature: "))
    humidity = float(input("Enter Humidity: "))
    ph = float(input("Enter pH: "))
    rainfall = float(input("Enter Rainfall: "))

    # Create a DataFrame from the input values
    sample = pd.DataFrame({
        "N": [N],
        "P": [P],
        "K": [K],
        "temperature": [temperature],
        "humidity": [humidity],
        "ph": [ph],
        "rainfall": [rainfall]
    })

    prediction = pipeline.predict(sample)
    print(f"Predicted Class: {prediction[0]}")


if __name__ == "__main__":
    get_prediction()