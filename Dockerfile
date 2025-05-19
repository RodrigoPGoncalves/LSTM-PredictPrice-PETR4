FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 5000
EXPOSE 8501
EXPOSE 8000

CMD ["python", "__init__.py"]
