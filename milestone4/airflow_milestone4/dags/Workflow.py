from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow import DAG

from green_data_tasks import create_clean_dataset
from green_data_tasks import extract_addtional_features
from green_data_tasks import load_to_postgres
from green_data_tasks import create_dashboard
import warnings
warnings.filterwarnings("ignore")

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 1,
}

dag = DAG(
    'green_tripdata_etl_pipeline',
    default_args=default_args,
    description='green_tripdata etl pipeline',
)
with DAG(
    dag_id = 'green_tripdata_etl_pipeline',
    schedule_interval = '@once',
    default_args = default_args,
    tags = ['green_tripdata-pipeline'],
)as dag:
    create_clean_dataset_task= PythonOperator(
        task_id = 'create_clean_dataset',
        python_callable = create_clean_dataset,
        op_kwargs={
            "file_name": 'green_tripdata_2019-08.csv',
            "data_dir": '/opt/airflow/data/'
        },
    )
    extract_addtional_features_task= PythonOperator(
        task_id = 'extract_addtional_features',
        python_callable = extract_addtional_features,
        op_kwargs={
            "file_name": 'green_trip_data_2019-8_clean.csv',
            "data_dir": '/opt/airflow/data/'
        },
    )
    load_to_postgres_task=PythonOperator(
        task_id = 'load_to_postgres',
        python_callable = load_to_postgres,
        op_kwargs={
            "file_name": "green_trip_data_2019-8_clean_with_features.csv",
            "lookup_name":"lookup_table_green_taxis.csv",
            "data_dir": '/opt/airflow/data/'
        },
    )
    create_dashboard_task= PythonOperator(
        task_id = 'create_dashboard_task',
        python_callable = create_dashboard,
        op_kwargs={
            "filename": "/opt/airflow/data/green_trip_data_2019-8_clean_with_features.csv"
        },
    )
    
    create_clean_dataset_task >> extract_addtional_features_task>> load_to_postgres_task >> create_dashboard_task

    
    



