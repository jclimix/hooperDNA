import requests
from bs4 import BeautifulSoup
import pandas as pd

# Step 1: Get the HTML content of the page
url = 'https://www.sports-reference.com/cbb/players/caitlin-clark-1.html'
response = requests.get(url)
html_content = response.content

# Step 2: Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Step 3: Find the element with the id 'players_per_game' (regardless of tag type)
element = soup.find(id='players_per_game')

# Check if the element was found
if element is None:
    print("The element with id 'players_per_game' was not found on the page.")
else:
    # Step 4: Extract the table from within the element (assuming it is a table or contains a table)
    table = element.find('table')

    # Step 5: Check if the table was found
    if table is None:
        print("The table within the 'players_per_game' element was not found.")
    else:
        # Step 6: Use Pandas to read the HTML table into a DataFrame
        df = pd.read_html(str(table))[0]

        # Step 7: Display the DataFrame
        print(df)
