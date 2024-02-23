from joblib import load
from eda import *


def extract_features(file_name: str, base_path: str = "NLP_interview_docs"):
    file_size = get_file_size(file_name)
    page_count = get_page_count(file_name)
    return [file_size, page_count]


def classify_document(file_name: str, base_path: str = "NLP_interview_docs"):
    loaded_model = load("decision_tree_model.joblib")
    loaded_label_encoder = load("label_encoder.joblib")

    new_features = extract_features(file_name, base_path)

    new_features_reshaped = np.array(new_features).reshape(
        1, -1
    )  # Reshape for a single sample
    predicted_class_index = loaded_model.predict(new_features_reshaped)[0]
    predicted_class_label = loaded_label_encoder.inverse_transform(
        [predicted_class_index]
    )
    return predicted_class_label[0]


if __name__ == "__main__":
    label = classify_document("PH-ME-P-0004-001.pdf")
    print(f"Predicted class for the new document: {label}")
