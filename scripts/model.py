import pandas as pd 
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib


def create_cluster(diabetes, nb_cluster=3):
    kmeans = KMeans(n_clusters=nb_cluster, random_state=42).fit(diabetes)
    diabetes['cluster_labels'] = kmeans.labels_
    return diabetes

def benchmark(diabetes, target_name='cluster_labels'):
    X = diabetes.drop(target_name, axis=1)
    y = diabetes[target_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define models
    models = [
        ("Logistic Regression", LogisticRegression()),
        ("Decision Tree", DecisionTreeClassifier()),
        ("Random Forest", RandomForestClassifier()),
        ("Support Vector Machine", SVC()),
        ("K-Nearest Neighbors", KNeighborsClassifier())
    ]

    # Train and evaluate models
    results = []
    best_model = None
    best_accuracy = 0

    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Classification Report": classification_rep
        })

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    # Convert results to a pandas DataFrame for better visualization
    results_df = pd.DataFrame(results)
    print(results_df)

    # Return the best model
    return best_model

def save_model(model, filename='best_model.pkl'):
    # Save the model to a file
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")
    






    