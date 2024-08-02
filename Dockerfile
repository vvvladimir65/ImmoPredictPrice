# Starts from the python 3.10 official docker image
FROM python:3.10

# Create a folder "app" at the root of the image
RUN mkdir /stremlitPlotMyReg

# Define /app as the working directory
WORKDIR /stremlitPlotMyReg

# Copy all the files in the current directory in /app
COPY . /stremlitPlotMyReg

# Update pip
RUN pip install --upgrade pip

# Install dependencies from "requirements.txt"
RUN pip install -r requirements.txt

# Run the Streamlit app
CMD ["streamlit", "run", "stremlitPlotMyReg.py", "--server.port=8501", "--server.address=0.0.0.0"]