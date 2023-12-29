# NYC Taxis Trips Analysis

This repository documents a data engineering project focusing on New York City's green and yellow taxi trips throughout August 2019. The project is divided into four distinct milestones, each addressing specific aspects of data processing, analysis, and automation.
## Motivation:

The motivation behind this project stems from the desire to explore and understand the intricacies of New York City's taxi data, specifically focusing on the green and yellow taxi trips during August 2019. By undertaking this project, we aim to:

- Gain insights into the trends, patterns, and behaviors exhibited within the taxi transportation system.
- Showcase the power of various data engineering tools such as Python, Docker, PostgreSQL, PySpark, Apache Airflow, and Python Dash in handling, processing, and visualizing vast amounts of data.
- Demonstrate the importance of reproducibility, portability, and automation in data engineering workflows, ensuring consistent and scalable analyses.
- Provide a resource that serves as both a learning opportunity for others interested in data engineering and a practical demonstration of applying these technologies in a real-world scenario.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/karim-walid-wahdan/NYC_Taxis_Trips_Analysis.git
   ```
2. Navigate to the respective milestone directory.
3. run the necessary software (Docker and/or PYSPARK).
4. run the necessary yaml file.
## Repository Structure:

- `milestone_1/`: Contains code and documentation for Milestone 1.
- `milestone_2/`: Includes files and configurations for Milestone 2.
- `milestone_3/`: Holds scripts and processes related to Milestone 3.
- `milestone_4/`: Contains automation scripts and dashboard creation files for Milestone 4.

## Milestones:

### Milestone 1: Exploratory Data Analysis (EDA) and Data Transformations

- **Objective**: Perform EDA and clean the dataset for green taxis (500,000 records).
- **Tools Used**: Python for data transformations and Jupyter Notebook for documentation.
- **Description**: Detailed exploration and cleaning techniques applied to the green taxi dataset, documented comprehensively in a Jupyter notebook.

### Milestone 2: Dockerization of the Process

- **Objective**: Containerize the entire data processing pipeline.
- **Tools Used**: Docker for containerization and PostgreSQL for data storage (volume mounted).
- **Description**: Docker configurations and setup instructions to ensure portability and reproducibility. Integration of the cleaned dataset into a PostgreSQL database with volume mounting for persistent storage.

### Milestone 3: Big Data Handling with PySpark

- **Objective**: Demonstrate PySpark's ability to handle large datasets.
- **Tools Used**: PySpark for analysis and cleaning (6 million records).
- **Description**: Utilization of PySpark to process and analyze the yellow taxi dataset. Containerization of the PySpark process for efficient handling of substantial data volumes.

### Milestone 4: Automation and Dashboard Creation

- **Objective**: Automate the data processing pipeline and create informative dashboards.
- **Tools Used**: Apache Airflow for automation and Python Dash for dashboard creation.
- **Description**: Implementation of Apache Airflow for automating the entire pipeline. Additionally, creation of insightful dashboards using Python Dash to visualize analyzed data.


## Contributing

Contributions to the NYC Taxis Trips Analysis project are welcome. If you find any issues or have suggestions for improvements, please create an issue or submit a pull request.
