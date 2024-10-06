from dotenv import load_dotenv
import os
import boto3
from botocore.exceptions import NoCredentialsError
import logging
import pandas as pd
import io

# Load environment variables from .env file
load_dotenv()

aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region = os.getenv('AWS_REGION')

s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=region
)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Function to read a CSV from S3 into a pandas DataFrame and print the first two rows
def read_csv_from_s3_with_pandas(bucket_name, object_name):
    """
    Read a CSV file from an S3 bucket into a pandas DataFrame and print the first two rows.

    Parameters:
    - bucket_name: The name of the S3 bucket.
    - object_name: The name of the file in the S3 bucket (e.g., 'myfile.csv').
    """
    
    try:
        # Get the CSV file from S3 as an object
        csv_object = s3.get_object(Bucket=bucket_name, Key=object_name)
        
        # Read the file content into memory (in bytes)
        csv_content = csv_object['Body'].read()
        
        # Use pandas to read the CSV content from the bytes object
        df = pd.read_csv(io.BytesIO(csv_content))
        
        # Print the first two rows of the DataFrame
        print(df.head(2))
        
    except NoCredentialsError:
        logging.error("Credentials not available")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

# Example usage
# Replace 'your-bucket-name' with your actual S3 bucket name
# Replace 's3-object-name.csv' with the name of the file in the S3 bucket

read_csv_from_s3_with_pandas('hooperdna-storage', 'college_basketball_players.csv')
