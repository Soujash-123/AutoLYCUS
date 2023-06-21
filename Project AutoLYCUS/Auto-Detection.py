import csv
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Read encrypted sentences and encryption names from CSV file
data = []
with open('encrypted_sentences.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header row
    for row in reader:
        data.append(row)

# Split data into features (encrypted strings) and labels (encryption names)
features = [row[2] for row in data]  # Encrypted strings
key_lengths = [len(row[1]) for row in data]  # Key lengths
labels = [row[3] for row in data]  # Encryption names

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create a pipeline to combine feature extraction and model training
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('scaler', StandardScaler(with_mean=False)),  # Scale key lengths
    ('classifier', RandomForestClassifier())
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
accuracy = pipeline.score(X_test, y_test)
print("Accuracy:", accuracy)
