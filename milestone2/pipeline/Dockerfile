FROM python:3.11

RUN pip install numpy
RUN pip install pandas
RUN pip install scipy
RUN pip install scikit-learn
RUN pip install requests   
RUN pip install sqlalchemy psycopg2

WORKDIR /app

ENTRYPOINT ["python","/app/src/main.py"]






