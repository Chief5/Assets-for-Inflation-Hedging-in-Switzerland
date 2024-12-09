# Use the official Python base image with the desired version
FROM python:3.12.7

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the code files into the container
COPY . .

# Expose a port (if your script has a web interface, e.g., Flask, or Streamlit)
# EXPOSE 8501

# Command to run your main Python script (adjust "main.py" to your entry script)
CMD ["python"]
