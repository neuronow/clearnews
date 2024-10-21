"""
This module provides parsers for various Polish news websites.

Each parser class inherits from the abstract base class PageParser and implements methods to:
- Retrieve article titles from the website's main page.
- Retrieve the content of an article given its URL.
- Retrieve the main image URL of an article.

The module includes parsers for the following news websites:
- TV Republika
- Do Rzeczy
- Gazeta.pl
- Najwyższy Czas (nczas.com)
- TVN24
- wPolityce.pl

It also provides a mapping of source names to their parser instances for easy access.
"""

from abc import ABC, abstractmethod
import logging
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup, Tag
from requests.exceptions import RequestException
from urllib.parse import urljoin, urlparse, urlunparse


def normalize_url(base_url: str, link_url: str) -> str:
    """
    Normalize and combine a base URL with a link URL.

    Args:
        base_url (str): The base URL to which the link URL is relative.
        link_url (str): The link URL, which may be relative or absolute.

    Returns:
        str: A normalized URL, combined from base_url and link_url,
             with query parameters and fragments removed.
    """
    combined_url = urljoin(base_url, link_url)
    parsed_url = urlparse(combined_url)
    normalized_url = urlunparse(
        (parsed_url.scheme, parsed_url.netloc, parsed_url.path, '', '', '')
    )
    return normalized_url


class PageParser(ABC):
    """
    Abstract base class for parsing news website pages.

    Subclasses must provide CSS selectors or attributes for locating
    article titles, article content, and article images.
    """

    article_content_selector: str = ""
    article_image_selector: str = ""
    image_src_attribute: str = "src"

    def __init__(self, url: str, name: str):
        """
        Initialize the PageParser with a base URL and a name.

        Args:
            url (str): The base URL of the news website.
            name (str): A unique name identifier for the news source.
        """
        self.url = url
        self.name = name

    def get_content(self, link: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse the HTML content of a given link.

        Args:
            link (str): The URL to fetch.

        Returns:
            Optional[BeautifulSoup]: Parsed HTML content if successful, None otherwise.
        """
        try:
            response = requests.get(link)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except RequestException as e:
            logging.error(f"Failed to retrieve content from {link}: {e}")
            return None

    def get_article_titles(self) -> List[Dict[str, str]]:
        """
        Retrieve a list of article titles from the news website.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing article titles,
                                  URLs, and source names.
        """
        content = self.get_content(self.url)
        if content is None:
            return []

        topics = []
        for link in content.find_all("a", href=True):
            title = self.extract_title(link)
            if title:
                url = normalize_url(self.url, link["href"])
                topics.append({"title": title, "url": url, "source": self.name})
        return topics

    @abstractmethod
    def extract_title(self, link: Tag) -> Optional[str]:
        """
        Extract the title from a link element.

        Args:
            link (Tag): The anchor tag from which to extract the title.

        Returns:
            Optional[str]: The title text if found, None otherwise.
        """
        pass

    def get_article(self, link: str) -> str:
        """
        Retrieve the content of an article given its link.

        Args:
            link (str): The URL of the article.

        Returns:
            str: The text content of the article.
        """
        content = self.get_content(link)
        if content is None:
            return ""
        article = content.select_one(self.article_content_selector)
        if article:
            return article.get_text(strip=True)
        logging.error(f"Article content not found for link: {link}")
        return ""

    def get_article_image(self, link: str) -> Optional[str]:
        """
        Retrieve the image URL of an article given its link.

        Args:
            link (str): The URL of the article.

        Returns:
            Optional[str]: The URL of the article's image, or None if not found.
        """
        content = self.get_content(link)
        if content is None:
            return None
        image_element = content.select_one(self.article_image_selector)
        if image_element and image_element.get(self.image_src_attribute):
            # Handle 'srcset' attribute if necessary
            image_url = image_element[self.image_src_attribute]
            if self.image_src_attribute == 'srcset':
                image_url = image_url.split()[0]
            return normalize_url(self.url, image_url)
        logging.error(f"Article image not found for link: {link}")
        return None


class TVRepublika(PageParser):
    """
    Parser for the TV Republika news website.
    """

    article_content_selector = "div.main-column"
    article_image_selector = "div.main-column img.media__element.img-fluid"
    image_src_attribute = "src"

    def __init__(self):
        """
        Initialize the TVRepublika parser with its base URL and name.
        """
        super().__init__(url="https://tvrepublika.pl", name="tvrepublika_pl")

    def extract_title(self, link: Tag) -> Optional[str]:
        """
        Extract the title from a link element for TV Republika.

        Args:
            link (Tag): The anchor tag from which to extract the title.

        Returns:
            Optional[str]: The title text if found, None otherwise.
        """
        title = link.get_text(strip=True)
        if not title:
            href_parts = link["href"].split("/")
            if len(href_parts) > 1:
                title = href_parts[-2].replace("-", " ")
            else:
                title = link["href"]
        return title if title else None


class DoRzeczyPL(PageParser):
    """
    Parser for the Do Rzeczy news website.
    """

    article_content_selector = "div#art-text-inner"
    article_image_selector = "picture.art-image-inner img"
    image_src_attribute = "src"

    def __init__(self):
        """
        Initialize the DoRzeczyPL parser with its base URL and name.
        """
        super().__init__(url="https://www.dorzeczy.pl/", name="dorzeczy_pl")

    def extract_title(self, link: Tag) -> Optional[str]:
        """
        Extract the title from a link element for Do Rzeczy.

        Args:
            link (Tag): The anchor tag from which to extract the title.

        Returns:
            Optional[str]: The title text if found, None otherwise.
        """
        return link.get("title", "").strip() or None


class GazetaPL(PageParser):
    """
    Parser for the Gazeta.pl news website.
    """

    article_content_selector = "div#gazeta_article_body"
    article_image_selector = "div.top_section img"
    image_src_attribute = "src"

    def __init__(self):
        """
        Initialize the GazetaPL parser with its base URL and name.
        """
        super().__init__(url="https://www.gazeta.pl/", name="gazeta_pl")

    def extract_title(self, link: Tag) -> Optional[str]:
        """
        Extract the title from a link element for Gazeta.pl.

        Args:
            link (Tag): The anchor tag from which to extract the title.

        Returns:
            Optional[str]: The title text if found, None otherwise.
        """
        return link.get("title", "").strip() or None


class NaCzasPL(PageParser):
    """
    Parser for the Najwyższy Czas news website (nczas.com).
    """

    article_content_selector = "div.td-post-content"
    article_image_selector = "div.tdb-block-inner img"
    image_src_attribute = "src"

    def __init__(self):
        """
        Initialize the NaCzasPL parser with its base URL and name.
        """
        super().__init__(url="https://www.nczas.com/", name="nczas_pl")

    def extract_title(self, link: Tag) -> Optional[str]:
        """
        Extract the title from a link element for nczas.com.

        Args:
            link (Tag): The anchor tag from which to extract the title.

        Returns:
            Optional[str]: The title text if found, None otherwise.
        """
        return link.get("title", "").strip() or None


class Tvn24PL(PageParser):
    """
    Parser for the TVN24 news website.
    """

    article_content_selector = "div.article-story-content"
    article_image_selector = "div.article-story-content img[srcset]"
    image_src_attribute = "srcset"

    def __init__(self):
        """
        Initialize the Tvn24PL parser with its base URL and name.
        """
        super().__init__(url="https://tvn24.pl", name="tvn24_pl")

    def extract_title(self, link: Tag) -> Optional[str]:
        """
        Extract the title from a link element for TVN24.

        Args:
            link (Tag): The anchor tag from which to extract the title.

        Returns:
            Optional[str]: The title text if found, None otherwise.
        """
        title = link.get_text(strip=True)
        return title if title else None


class WPolitycePL(PageParser):
    """
    Parser for the wPolityce.pl news website.
    """

    article_content_selector = "section.article__main"
    article_image_selector = "img.article-img__img"
    image_src_attribute = "src"

    def __init__(self):
        """
        Initialize the WPolitycePL parser with its base URL and name.
        """
        super().__init__(url="https://wpolityce.pl", name="wpolityce_pl")

    def extract_title(self, link: Tag) -> Optional[str]:
        """
        Extract the title from a link element for wPolityce.pl.

        Args:
            link (Tag): The anchor tag from which to extract the title.

        Returns:
            Optional[str]: The title text if found, None otherwise.
        """
        title = link.get_text(strip=True)
        return title if title else None


# Instantiate the news sources
news_sources = [
    TVRepublika(),
    DoRzeczyPL(),
    GazetaPL(),
    Tvn24PL(),
    WPolitycePL(),
    NaCzasPL(),
]

# Create a mapping of source names to their instances
news_mapping = {source.name: source for source in news_sources}