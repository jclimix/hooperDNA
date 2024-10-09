# HooperDNA: College Basketball and NBA Player Comparison

This web application scrapes player data from College Basketball and NBA sources (Sports Reference, Basketball Reference) and compares statistics between players. The app provides users with a simple interface to input player names or IDs, and it outputs a side-by-side comparison.

## Table of Contents
1. [Features](#features)
2. [Technologies Used](#technologies-used)
3. [Setup and Installation](#setup-and-installation)
4. [Usage](#usage)
5. [Docker Instructions](#docker-instructions)
6. [Environment Variables](#environment-variables)
7. [Known Issues](#known-issues)
8. [Contributing](#contributing)
9. [License](#license)

## Features

- Scrapes player data from Sports Reference and Basketball Reference.
- Compares College Basketball and NBA players' statistics.
- Interactive web interface for user input and comparison.
- Uses AWS S3 for data storage and retrieval.
- Dockerized for easy deployment.

## Technologies Used

- Python
- Flask
- BeautifulSoup
- Pandas
- Boto3
- Docker
- AWS S3