# $ vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
#     --host 0.0.0.0 \
#     --port 8000 \
#     --limit-mm-per-prompt '{"image": 5}' \
#     --max-model-len 32768 # Use a more realistic context length


import requests
import json

# Define the server endpoint
api_url = "http://localhost:8000/v1/chat/completions"

# Define the request payload
payload = {
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://media.roboflow.com/notebooks/examples/dog.jpeg"}},
                {"type": "text", "text": "Describe this image."}
            ]
        }
    ],
    "max_tokens": 1024,
    "stream": False
}

# Send the request
headers = {"Content-Type": "application/json"}
response = requests.post(api_url, headers=headers, data=json.dumps(payload))

# Print the response
if response.status_code == 200:
    print(response.json()['choices'][0]['message']['content'])
else:
    print(f"Error: {response.status_code}, {response.text}")
