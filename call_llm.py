import os
import re
import json
import requests
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.llms.base import LLM


# Custom LLM class for OpenRouter
class CustomLLM(LLM):
    def _call(self, prompt, stop=None):
        messages = [{"role": "user", "content": prompt}]
        return generate_response(messages)

    @property
    def _identifying_params(self):
        return {"model_name": "qwen/qwen-2-7b-instruct"}

    @property
    def _llm_type(self):
        return "custom"

# Function to generate a response using the OpenRouter API
def generate_response(messages):
     
    OPENROUTER_API_KEY= os.environ.get("OPENROUTER_API_KEY")
    endpoint_url = "https://openrouter.ai/api/v1/chat/completions"

    # Prepare the payload
    payload = {
        "model": "qwen/qwen-2-7b-instruct",
        "messages": messages,
        "max_tokens": 1500,
        "temperature": 0.3,
        "stream": True
    }

    # Send the request to OpenRouter API
    response = requests.post(
        url=endpoint_url,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        data=json.dumps(payload),
        stream=True
    )

    # Check for response status code
    if response.status_code != 200:
        raise ValueError(f"API request failed with status code {response.status_code}: {response.text}")

    # Process the streaming response
    output = ""
    for chunk in response.iter_lines(decode_unicode=True):
        if not chunk:
            continue  # Skip empty lines

        # Handle chunks with "data: " prefix
        if chunk.startswith("data: "):
            chunk = chunk[len("data: "):]  # Remove the "data: " prefix

        try:
            # Parse the chunk as JSON
            data = json.loads(chunk)
            # Extract content if available
            if 'choices' in data and 'delta' in data['choices'][0] and 'content' in data['choices'][0]['delta']:
                cleaned_content = re.sub(r'\s+', ' ', data['choices'][0]['delta']['content'])
                output += cleaned_content
        except json.JSONDecodeError:
            print(" ")  # Log invalid chunks
            continue

    return output


#if __name__ == "__main__":
    # Initialize the custom LLM
#    llm = CustomLLM()
