# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container at /app
COPY . /app/

# Expose the port that the gRPC server will use
EXPOSE 50052

# Command to run the server (replace 'server.py' with the actual entry point to start your gRPC server)
CMD ["python", "NLP/PC0/server.py"]
