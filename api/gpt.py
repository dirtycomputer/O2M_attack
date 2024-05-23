import requests
from apikeys import apikeys
from util import encode_image

# OpenAI API Key
api_key = apikeys.openai

def gpt(image_path, text, model="gpt-4o", max_tokens=300):
    assert model in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": model,
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": text
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": max_tokens
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response.json()["choices"][0]["message"]["content"]