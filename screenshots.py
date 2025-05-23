import requests
from bs4 import BeautifulSoup
import os
import pandas as pd


class Screenshots:
    """
    A class to scrape and save screenshots of Steam games using their app IDs.

    Attributes:
        file_path (str): The path to the CSV file containing game data.
        start_row (int): The starting index in the CSV to begin scraping.
        df (pd.DataFrame): The loaded DataFrame from the CSV.
    """

    def __init__(self, file_path, start_row):
        """
        Initializes the Screenshots instance with the CSV file and start row.

        Args:
            file_path (str): Path to the CSV file.
            start_row (int): The row index to start scraping from.
        """
        self.file_path = file_path
        self.start_row = start_row
        self.df = pd.read_csv(file_path)
        os.makedirs("img", exist_ok=True)


    def scrape(self, number):
        """
        Scrapes screenshots for a specified number of rows starting from start_row.

        Args:
            number (int): The number of screenshots to scrape.
        """
        for index, row in self.df.iloc[self.start_row:self.start_row + number].iterrows():
            aid = row['app_id']
            self.scrape_single_img(aid)  # Fixed from self.scrape(aid)
            self.start_row += 1

    def scrape_single_img(self, app_id):
        """
        Scrapes and saves a single screenshot for a given Steam app ID.

        Args:
            app_id (int or str): Steam App ID of the game.
        """
        url = f"https://store.steampowered.com/app/{app_id}/"
        headers = {"User-Agent": "Mozilla/5.0"}

        # Get HTML content
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        save_dir = "img"
        os.makedirs(save_dir, exist_ok=True)

        # Extract and save screenshot
        screenshot = soup.find('a', class_='highlight_screenshot_link')
        if screenshot:
            img_url = screenshot['href']
            img_response = requests.get(img_url, headers=headers)

            if img_response.status_code == 200:
                img_path = os.path.join(save_dir, f"{app_id}.jpg")
                with open(img_path, 'wb') as f:
                    f.write(img_response.content)
                print(f"Saved {img_path}")
            else:
                print(f"Failed to download {img_url}")
        else:
            print(f"No screenshot found for app ID: {app_id}")




# Example usage
fp = "games.csv"

# change starting row
starting_row = 0
ss = Screenshots(fp, starting_row)

# change number of images to scrape
number = 1
ss.scrape(number)
