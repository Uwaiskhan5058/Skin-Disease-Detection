FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models uploads

# Expose port (Render sets its own dynamically as well)
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application using shell form so $PORT resolves correctly
CMD gunicorn app.main:app --bind 0.0.0.0:${PORT:-5000} --workers 1 --timeout 120 --preload
