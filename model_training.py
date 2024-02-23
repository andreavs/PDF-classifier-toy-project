import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from joblib import dump
from eda import *


if __name__ == "__main__":
    random_state = 44  # conveniently making sure that all the document types are represented in the training set
    data = labeled_data()

    add_feature(data, "file_size", get_file_size)
    add_feature(data, "page_count", get_page_count)

    # Assuming 'data' is your dataset as described previously
    features = np.array(
        [[doc_data["file_size"], doc_data["page_count"]] for doc_data in data.values()]
    )
    labels = np.array([doc_data["class"] for doc_data in data.values()])

    # Encoding the class labels to integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels_encoded, test_size=0.2, random_state=random_state
    )
    print(X_test, X_train, y_train, y_test)

    dt_classifier = DecisionTreeClassifier(random_state=random_state)
    dt_classifier.fit(X_train, y_train)

    # Calculating the accuracy
    y_pred = dt_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Save the model to a file
    model_filename = "decision_tree_model.joblib"
    dump(dt_classifier, model_filename)
    encoder_filename = "label_encoder.joblib"
    dump(label_encoder, encoder_filename)
