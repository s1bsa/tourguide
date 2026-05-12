FROM python:3.11-slim

WORKDIR /database

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . . 

CMD ["uvicorn", "db_manager:app", "--host", "0.0.0.0", "--port", "7000"]
