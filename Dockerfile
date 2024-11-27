FROM python:3.7


COPY . /app
RUN pip install -r requirements.txt

WORKDIR /app

EXPOSE 7860

CMD ["python", "app.py"]