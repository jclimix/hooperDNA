import pandas as pd
import random
import numpy as np
import requests
import re
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import requests
from bs4 import BeautifulSoup
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def scrape_college_headshot(id):

    url = f'https://www.sports-reference.com/cbb/players/{id}.html'
    
    # Step 1: Fetch the HTML content of the page
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text
    else:
        logger.error("Failed to retrieve the webpage. Status code: %s", response.status_code)
        return "https://i.ibb.co/vqkzb0m/temp-player-pic.png"  # Return default image link

    # Step 2: Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")
    target_id = "meta"
    target_section = soup.find(id=target_id)
    college_image_link = "https://i.ibb.co/vqkzb0m/temp-player-pic.png"  # Default image link

    # Step 3: Search for the image within the "meta" section
    if target_section:
        media_items = target_section.find_all("div", class_="media-item")

        if media_items:
            for index, item in enumerate(media_items, 1):
                img_tag = item.find("img")
                if img_tag and "src" in img_tag.attrs:
                    college_image_link = img_tag["src"]
                    logger.info(f"College player headshot link found: {college_image_link}")
                    break
                else:
                    logger.error(f"Image {index}: No image found.")
        else:
            logger.error(f"No media items found in the section with id '{target_id}'.")
    else:
        logger.error(f"No section found with id '{target_id}'.")

    return college_image_link

college_id = 'zach-edey-1'
link = scrape_college_headshot(college_id)
print(link)
