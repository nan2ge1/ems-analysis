# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the file with the requirements necessary to run the app
COPY requirements.txt .
COPY pyproject.toml .

# Install packages specified in requirements.txt
# We install directly from requirements.txt to keep the image small
# and avoid installing dev dependencies unless needed.
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install .

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app.py when the container launches
# --server.port=8501: Listen on port 8501
# --server.address=0.0.0.0: Listen on all interfaces (required for Docker)
CMD ["streamlit", "run", "src/ems/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
