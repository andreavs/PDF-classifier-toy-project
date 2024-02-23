import base64
import requests
import os
from pdf2image import convert_from_path
import numpy as np
import PIL 

MAX_IMAGE_SIZE=10_000_000

def encode_image(image_path):
    """base 64 encode image"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def ask_chatgpt(api_key, file_name, base_path="NLP_interview_docs/"):
    PIL.Image.MAX_IMAGE_PIXELS = 10*200213712
    image_path = "image.png"
    full_path = os.path.join(base_path, file_name)

    # create png of first page:
    images = convert_from_path(full_path, last_page=1)
    image = images[0]

    pixels = image.height*image.width
    if pixels >= MAX_IMAGE_SIZE: # upper limit for gpt
        ratio = MAX_IMAGE_SIZE/pixels
        image = image.resize((int(image.width*np.sqrt(ratio)-1), int(image.height*np.sqrt(ratio) - 1)))
    image.save(image_path, "PNG")

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    prompt = "I want you to classify this image as one of the following: engineering_diagram, data_sheet, instruction_manual, or other. Respond with the class, nothing else"

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    print(response.json())
    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    from eda import labeled_data
    api_key = os.environ["OPENAI_API_KEY"]
    data = labeled_data()
    for file, label in data.items(): 
        label_by_gpt = ask_chatgpt(api_key, file)
        print(f"{file}: {label['class']} - {label_by_gpt}")

