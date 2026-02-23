import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Preprocess the data."""
    # Example preprocessing steps
    # E.g., handle missing values, encode categorical variables
    data.fillna(0, inplace=True)
    # Perform encoding and scaling as needed
    return data

def train_model(X_train, y_train):
    """Train the classification model."""
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True)
    plt.title("Confusion Matrix")
    plt.show()

def main():
    """Main function to run the ML pipeline."""
    file_path = 'legal_cases_extended.csv'
    data = load_data(file_path)
    data = preprocess_data(data)

    # Assuming the target variable is named 'target'
    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()