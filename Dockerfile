FROM python:3.7

RUN pip install gradio litellm

COPY . /app

WORKDIR /app

EXPOSE 7860

CMD ["python", "app.py"]