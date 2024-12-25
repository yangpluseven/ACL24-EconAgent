import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import seaborn as sns
import re
import os
import multiprocessing
import scipy
import requests
import json

# Change this to your own path and model if you want
save_path = "./"
ollama_model = "llama3.1"

brackets = list(np.array([0, 97, 394.75, 842, 1607.25, 2041, 5103]) * 100 / 12)
quantiles = [0, 0.25, 0.5, 0.75, 1.0]

from datetime import datetime

world_start_time = datetime.strptime("2001.01", "%Y.%m")

prompt_cost_1k, completion_cost_1k = 0.001, 0.002


def chat_using_ollama(messages, model=ollama_model, num_predict=2048, temperature=0):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": temperature},
    }

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        try:
            lines = response.text.strip().split("\n")
            last_response = json.loads(lines[-1])
            return last_response.get("message", {}).get("content", "")
        except Exception as e:
            return f"Error parsing response: {str(e)}"
    else:
        return f"Error: {response.status_code}"


def get_ollama_completion(dialogs, model=ollama_model, num_predict=2048, temperature=0):
    max_retries = 3
    for i in range(max_retries):
        try:
            response = chat_using_ollama(
                dialogs, model=model, num_predict=num_predict, temperature=temperature
            )
            return response
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(6)
            else:
                print(f"An error of type {type(e).__name__} occurred: {e}")
                return "Error"


def get_multiple_ollama_completion(dialogs, model="llama3.1", num_predict=2048, temperature=0):
    results = []
    for dialog in dialogs:
        results.append(
            get_ollama_completion(
                dialog, model=model, num_predict=num_predict, temperature=temperature
            )
        )
    return results


def prettify_document(document: str) -> str:
    # Remove sequences of whitespace characters (including newlines)
    cleaned = re.sub(r"\s+", " ", document).strip()
    return cleaned


def get_multiple_completion(dialogs, num_cpus=15, temperature=0, max_tokens=100):
    from functools import partial

    get_completion_partial = partial(
        get_completion, temperature=temperature, max_tokens=max_tokens
    )
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = pool.map(get_completion_partial, dialogs)
    total_cost = sum([cost for _, cost in results])
    return [response for response, _ in results], total_cost


def get_completion(dialogs, temperature=0, max_tokens=100):
    import openai

    openai.api_key = "Your Key"
    import time

    max_retries = 20
    for i in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",  # inaccessible now, try gpt-4o-mini
                messages=dialogs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            this_cost = (
                prompt_tokens / 1000 * prompt_cost_1k
                + completion_tokens / 1000 * completion_cost_1k
            )
            return response.choices[0].message["content"], this_cost
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(6)
            else:
                print(f"An error of type {type(e).__name__} occurred: {e}")
                return "Error"


def format_numbers(numbers):
    return "[" + ", ".join("{:.2f}".format(num) for num in numbers) + "]"


def format_percentages(numbers):
    return "[" + ", ".join("{:.2%}".format(num) for num in numbers) + "]"
