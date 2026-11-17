FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 CHATBOT_DB=/data/chatbot.db HOST=0.0.0.0 PORT=5000

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

RUN useradd -m bot && mkdir /data && chown bot /data
USER bot
VOLUME /data
EXPOSE 5000

# Set ADMIN_TOKEN at run time:  docker run -e ADMIN_TOKEN=... -p 5000:5000 -v chatdata:/data chatbot
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:create_app()"]
