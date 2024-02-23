import matplotlib.pyplot as plt
import numpy as np
import PyPDF2
from pdfminer.high_level import extract_text
import os


def labeled_data():
    return {
        "PH-25578-P-4110006-001.pdf": {"class": "engineering_diagram"},
        "PH-25578-P-4110010-001.pdf": {"class": "engineering_diagram"},
        "PH-25578-P-4110119-001.pdf": {"class": "engineering_diagram"},
        "PH-ME-P-0004-001.pdf": {"class": "engineering_diagram"},
        "PH-ME-P-0151-001.pdf": {"class": "engineering_diagram"},
        "PH-ME-P-0153-001.pdf": {"class": "engineering_diagram"},
        "PH-ME-P-0156-001.pdf": {"class": "engineering_diagram"},
        "PH-ME-P-0156-002.pdf": {"class": "engineering_diagram"},
        "27PSV__1035.pdf": {"class": "data_sheet"},
        "32-PSV-95809-SP.pdf": {"class": "data_sheet"},
        "45-PSV-92515-SP.pdf": {"class": "data_sheet"},
        "SRM00064.pdf": {"class": "instruction_manual"},
        "norsok-p-002_2023_en_003.pdf": {"class": "other"},
    }


def get_file_size(file_name: str, base_path: str = "NLP_interview_docs/"):
    full_path = os.path.join(base_path, file_name)

    # Get the file size in bytes
    file_size_bytes = os.path.getsize(full_path)
    # Convert the file size to kilobytes
    file_size_kb = file_size_bytes / 1024
    return file_size_kb


def get_page_count(file_name: str, base_path: str = "NLP_interview_docs/"):
    full_path = os.path.join(base_path, file_name)
    try:
        with open(full_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            num_pages = len(pdf_reader.pages)
        return num_pages
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        return None


def get_word_count(file_name: str, base_path: str = "NLP_interview_docs/"):
    full_path = os.path.join(base_path, file_name)
    word_count = 0
    try:
        text = extract_text(full_path)
        word_count = len(text.split())
        return word_count
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        return None


def add_feature(
    data, feature_name, feature_func, base_path: str = "NLP_interview_docs"
):
    for file_name in data:
        data[file_name][feature_name] = feature_func(file_name, base_path)


import matplotlib.pyplot as plt


def plot_feature(data, feature_name):
    # Preparing data for plotting
    categories = {
        doc_data["class"] for doc_data in data.values()
    }  # Extracting unique categories
    category_colors = plt.get_cmap("tab10")(
        np.linspace(0, 1, len(categories))
    )  # Assign colors
    color_map = dict(zip(categories, category_colors))  # Map categories to colors

    plt.figure(figsize=(10, 6))

    # Plot each document
    for idx, (file_name, doc_data) in enumerate(data.items()):
        category = doc_data["class"]
        feature_value = doc_data.get(feature_name)
        if feature_value is not None:  # Ensure the feature exists
            plt.scatter(
                idx, feature_value, color=color_map[category], label=category, alpha=0.7
            )

    # Creating the legend without duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Removing duplicates
    plt.legend(by_label.values(), by_label.keys())

    plt.title(f"{feature_name.capitalize()} by Document Index")
    plt.xlabel("Document Index")
    plt.ylabel(feature_name.capitalize())
    plt.yscale("log")  # Using logarithmic scale for y-axis
    plt.grid(True, which="both", ls="--")
    plt.show()


if __name__ == "__main__":
    data = labeled_data()
    add_feature(data, "file_size", get_file_size)
    add_feature(data, "page_count", get_page_count)
    add_feature(data, "word_count", get_word_count)

    # Plotting each feature
    plot_feature(data, "file_size")
    plot_feature(data, "page_count")
    plot_feature(data, "word_count")
