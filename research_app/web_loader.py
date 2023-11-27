import requests
from bs4 import BeautifulSoup

def scrape_text(url: str):
    # Send a GET request to the webpage
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the content of the request with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract all text from the webpage
            page_text = soup.get_text(separator=" ", strip=True)

            # Print the extracted text
            return page_text
        else:
            raise Exception(f"Failed to retrieve the webpage: Status code {response.status_code}")
    except Exception as e:
        print(e)
        raise Exception(f"Failed to retrieve the webpage: {e}")
