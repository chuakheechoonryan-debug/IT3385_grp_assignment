FROM python:3.10

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
COPY . /app

# Install dependencies
RUN pip install -U -r requirements.txt

# Expose port 
EXPOSE 5000

# Run the application:
CMD ["python", "src/assignment_combi_app.py"]