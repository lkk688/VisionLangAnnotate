FROM python:3.9
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/src /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]