# Use an official Python image as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy everything from the local repository to the container, except .venv (handled by .dockerignore)
COPY . .

# Install the dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to start the Streamlit server with main.py
CMD ["streamlit", "run", "main.py"]
