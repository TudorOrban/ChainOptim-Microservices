FROM python:3.10-slim

WORKDIR /code

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set DGL backend to PyTorch
ENV DGLBACKEND=pytorch

COPY ./app /code/app

# Ensure Python knows where to find your custom modules
ENV PYTHONPATH="${PYTHONPATH}:/code"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]