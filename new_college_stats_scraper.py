import requests
import re
from bs4 import BeautifulSoup
import pandas as pd

def scrape_college_data(id):
        
    # Step 1: Download the webpage and store the content in memory
    url = f'https://www.sports-reference.com/cbb/players/{id}.html'  # Replace with the actual URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Store the content directly in memory
        content = response.text
    else:
        print("Failed to retrieve the webpage. Status code:", response.status_code)
        exit()

    # Step 2: Use regex to find the JavaScript block containing the target div
    # The regex pattern searches for the div with id 'div_players_per_game' and captures the HTML content within it
    pattern = r'<div class="table_container tabbed current" id="div_players_per_game">(.*?)</div>'
    matches = re.search(pattern, content, re.DOTALL)

    # Check if the pattern was found
    if matches:
        # Extract the div's content (which includes the table HTML)
        div_content = matches.group(1)
        
        # Step 3: Parse the extracted HTML snippet to find the table
        soup = BeautifulSoup(div_content, 'html.parser')
        table = soup.find('table')
        
        if table:
            # Step 4: Convert the HTML table to a DataFrame
            df = pd.read_html(str(table))[0]  # [0] gets the first table

            df = df.iloc[[-2]]
            
            #print(df)
            return df
        
        else:
            print("Table not found within the extracted div content.")
    else:
        print("Div with id 'div_players_per_game' not found in the JavaScript.")

college_player_id = 'caitlin-clark-1'
college_data = scrape_college_data(college_player_id)
print(college_data)