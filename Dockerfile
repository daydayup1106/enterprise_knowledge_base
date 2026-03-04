FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy project files
COPY . /app/

# Create conda environment and install all dependencies
RUN conda create -n enterprise_knowledge_base python=3.11 -y && \
    conda run -n enterprise_knowledge_base pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8181

# Run using the conda environment
CMD ["conda", "run", "--no-capture-output", "-n", "enterprise_knowledge_base", \
     "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8181"]
