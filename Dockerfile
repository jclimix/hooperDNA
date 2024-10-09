# Use an official Python runtime as a parent image
FROM python:3.10.11

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the Flask app runs on
EXPOSE 8005

# Define environment variable
ENV FLASK_APP=app.py

# Run Gunicorn to serve the Flask app
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8005", "app:app"]
