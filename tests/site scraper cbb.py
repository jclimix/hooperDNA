import requests
from bs4 import BeautifulSoup
import pandas as pd

# Step 1: Get the HTML content of the page
url = 'https://www.sports-reference.com/cbb/players/caitlin-clark-1.html'
response = requests.get(url)
html_content = response.content

# Step 2: Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Step 3: Find the element with the id 'div_players_per_game'
div = soup.find('div', id='div_players_per_game')

# Step 4: Extract the table from within the div (assuming it is a table)
table = div.find('table')

# Step 5: Use Pandas to read the HTML table into a DataFrame
df = pd.read_html(str(table))[0]

# Step 6: Display the DataFrame
print(df)
