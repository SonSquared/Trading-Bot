FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY trading_system/ trading_system/
COPY configs/ configs/
COPY scripts/run_paper_trading.py scripts/

# State persistence
VOLUME /app/data

# Run the bot in continuous paper-trading mode
CMD ["python", "-u", "scripts/run_paper_trading.py", "--continuous"]
