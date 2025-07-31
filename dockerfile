FROM python:3.12.7-slim
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Correct typo in 'pip install'
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt

# Copy all remaining files into the container
COPY . .

# Expose port 5000
EXPOSE 5000

# Use double quotes and exec form for CMD, and ensure proper syntax
CMD ["flask", "--app", "app", "--debug", "run", "--host=0.0.0.0"]
