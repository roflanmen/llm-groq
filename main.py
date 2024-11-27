import gradio as gr
from litellm import completion
import os


def run(character, world, plot, api_key):
    os.environ['GROQ_API_KEY'] = api_key
    response = completion(
        model="groq/llama3-8b-8192", 
        messages=[
            {
                "role": "user", 
                "content": f"""
                    Imagine you are the master storyteller of an ancient kingdom. You have been summoned by the king to weave a grand tale based on the following elements: 
                    A character provided by the user: {character}
                    A setting provided by the user: {world}
                    A plot twist provided by the user: {plot}
                    Using these elements, craft a vivid and immersive story that surprises and delights the audience. Feel free to expand creatively on the details provided while maintaining the spirit of the user’s input
                """}
        ],
    )
    return response['choices'][0]['message']['content']
demo = gr.Interface(
    fn=run,
    inputs=["text", "text", "text", "text"],
    outputs=["text"],
)
demo.launch()