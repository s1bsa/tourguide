FROM python:3.11-slim 

WORKDIR /yolo_service 

COPY requirements.txt requirements.txt 

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* 

RUN pip install -r requirements.txt 

COPY . . 

CMD ["uvicorn", "yoloapi:app", "--host", "0.0.0.0", "--port", "8000"] 