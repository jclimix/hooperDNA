from dotenv import load_dotenv
import os
import boto3
from botocore.exceptions import NoCredentialsError
import logging

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

# Function to upload a file to an S3 bucket with a custom name
def upload_to_s3(file_path, bucket_name, object_name):
    """
    Upload a file to an S3 bucket.

    Parameters:
    - file_path: The local path to the file that you want to upload.
    - bucket_name: The name of the S3 bucket.
    - object_name: The custom name to save the file as in the S3 bucket.
    """
    
    if not os.path.isfile(file_path):
        logging.error(f"File {file_path} does not exist.")
        return False
    
    try:
        # Upload the file to S3 with the custom object name
        s3.upload_file(file_path, bucket_name, object_name)
        logging.info(f"File {file_path} uploaded to {bucket_name}/{object_name}")
        return True
    except FileNotFoundError:
        logging.error("The file was not found")
        return False
    except NoCredentialsError:
        logging.error("Credentials not available")
        return False
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return False

# Example usage
# Replace '/path/to/your/local/file.csv' with the path to your local CSV file
# Replace 'your-bucket-name' with your actual S3 bucket name
# Replace 'custom-s3-object-name.csv' with the name you want to use in the S3 bucket

upload_to_s3(
    '.\sample_DB\college_data\college_basketball_players.csv',
    'hooperdna-storage',
    'college_basketball_players.csv'
)
