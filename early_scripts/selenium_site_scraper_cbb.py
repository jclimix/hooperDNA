from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd

# Step 1: Set up Selenium with ChromeDriver in headless mode with optimizations
service = Service(executable_path='./misc/chromedriver-win64/chromedriver.exe')
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Headless mode
options.add_argument("--disable-extensions")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_experimental_option("prefs", {
    "profile.managed_default_content_settings.images": 2,
    "profile.managed_default_content_settings.stylesheets": 2,
})
driver = webdriver.Chrome(service=service, options=options)

# Step 2: Navigate to the page
url = 'https://www.sports-reference.com/cbb/players/tim-duncan-1.html'
driver.get(url)

# Step 3: Wait for the table to fully load without unnecessary delay
try:
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "players_per_game"))
    )
except Exception as e:
    print("Table load timeout or error:", e)

# Step 4: Parse the page content with BeautifulSoup
html_content = driver.page_source
soup = BeautifulSoup(html_content, 'html.parser')

# Step 5: Find the table element directly by ID
table = soup.find('table', id='players_per_game')

# Step 6: Check if the table is found and then use Pandas to convert it to a DataFrame
if table:
    df = pd.read_html(str(table))[0]
    print(df)
else:
    print("Table with id 'players_per_game' not found.")

# Step 7: Close the Selenium browser
driver.quit()
