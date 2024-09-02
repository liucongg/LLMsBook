"""
model deploy : faschat
  run:
    1.python -m fastchat.serve.controller
    2.python -m fastchat.serve.model_worker --model-path ./chatglm2-6b --num-gpus 2 --host=0.0.0.0 --port=21002

calling interface : requests.post
"""

import requests

import configs


def get_response(text):
    headers = {"Content-Type": "application/json"}
    pload = {
        "model": "chatglm2-6b",
        "prompt": text,
        "stop": "###",
        "max_new_tokens": 8000,
    }
    print("pload", pload)
    response = requests.post(configs.llm_service_url_address + "/worker_generate_stream", headers=headers, json=pload, stream=True)
    # print(response.text)
    return response
