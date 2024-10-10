# HooperDNA: College Basketball and NBA Player Comparison

This web application scrapes player data from College Basketball and NBA sources (Sports Reference, Basketball Reference) and compares statistics between players. The app provides users with a simple interface to input player names or IDs, and it outputs a side-by-side comparison. Additionally, automated scripts running on Airflow ensure that the datasets stored on AWS S3 are continuously updated, providing users with the most current data. An API, built using Django, also allows for programmatic access to the comparison tool.

## Features

- Scrapes player data from Sports Reference and Basketball Reference.
- Compares College Basketball and NBA players' statistics.
- Interactive web interface for user input and comparison.
- Uses AWS S3 for data storage and retrieval.
- Automated dataset updates hosted on Airflow.
- Provides an API built with Django for programmatic access to the comparison tool.
- Dockerized for easy deployment.

## Technologies Used

- Python
- Flask
- BeautifulSoup
- Pandas
- Boto3
- AWS S3
- Airflow (for automating data updates)
- Django (for building the API)
- Docker
