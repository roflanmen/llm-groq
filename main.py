import gradio as gr
from litellm import completion
import os


def run(персонаж, світ, сюжет, api_key):
    os.environ['GROQ_API_KEY'] = api_key
    response = completion(
        model="groq/llama3-8b-8192", 
        messages=[
            {
                "role": "user", 
                "content": f"""
                    Уявіть, що ви майстерний оповідач стародавнього королівства. Ви були викликані королем, щоб сплести грандіозну історію, засновану на таких елементах: 
                    Персонаж, наданий користувачем: {персонаж}
                    Налаштування, надані користувачем: {світ}
                    Сюжет, наданий користувачем: {сюжет}
                    Використовуючи ці елементи, створіть яскраву та захоплюючу історію, яка здивує та захопить аудиторію. Не соромтеся творчо розширювати надані деталі, зберігаючи дух введення користувача
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