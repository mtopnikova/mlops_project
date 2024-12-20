FROM python:3.11-slim-buster

WORKDIR /app/frontend
COPY . .

RUN pip install --upgrade pip && \
    pip install --ignore-installed -r requirements.txt


EXPOSE 8501

ENTRYPOINT ["streamlit","run"]
CMD ["main.py"]