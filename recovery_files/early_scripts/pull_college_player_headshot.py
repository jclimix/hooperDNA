import requests
from bs4 import BeautifulSoup

# Function to scrape the web page
def scrape_college_img(url, target_id):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the section with the specified id
        target_section = soup.find(id=target_id)

        if not target_section:
            print(f"No section found with id '{target_id}'.")
            return

        # Find all div elements with class 'media-item' within the target section
        media_items = target_section.find_all('div', class_='media-item')

        if media_items:
            # Loop through the media items and extract the image link
            for index, item in enumerate(media_items, 1):
                img_tag = item.find('img')
                if img_tag and 'src' in img_tag.attrs:
                    img_link = img_tag['src']
                    print(f"Image {index} link: {img_link}")
                else:
                    print(f"Image {index}: No image found.")
        else:
            print(f"No media items found in the section with id '{target_id}'.")

    except requests.exceptions.RequestException as e:
        print(f"Error while scraping the webpage: {e}")

# URL of the page you want to scrape
url = 'https://www.sports-reference.com/cbb/players/caitlin-clark-1.html'  # Replace with the actual URL

# The target id where the media items are located
target_id = 'meta'  # Replace with the actual id

# Call the function to scrape the page
scrape_college_img(url, target_id)
