FROM python:3.11.4-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install dependencies

# Install any necessary dependencies
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Download files from URLs
RUN wget https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth
RUN wget https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy project files into the container
COPY . /app/

# Expose port
EXPOSE 8000

# Command to run the Django app
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
# ENTRYPOINT [ "gunicorn", "core.wsgi", "-b", "0.0.0.0:8000"]