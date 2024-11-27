import gradio as gr
from litellm import completion
import os


def run(text, api_key):
    os.environ['GROQ_API_KEY'] = api_key
    response = completion(
        model="groq/llama3-8b-8192", 
        messages=[
            {"role": "user", "content": "hi. can you please translate this text to english, send only result text: " + text}
        ],
    )
    return response['choices'][0]['message']['content']
demo = gr.Interface(
    fn=run,
    inputs=["text", "text"],
    outputs=["text"],
)
demo.launch()