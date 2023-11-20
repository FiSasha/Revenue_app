FROM python:3.10.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -U pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD streamlit run app.py