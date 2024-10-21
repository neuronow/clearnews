import json
import logging
import os
import time
from itertools import compress
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm, trange

from sources import PageParser, news_mapping, news_sources

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load the API key from an environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")
# Pass the API key directly when initializing the models

NEWS_CATEGORIES = [
    "politics", "technology", "science",
    "economy", "war", "sport", "other",
    "culture", "health", "education"
]


class WouldClick(BaseModel):
    would_click: bool = Field("True or false value")


class WouldClickList(BaseModel):
    would_click_list: List[WouldClick] = Field(default_factory=list)


class NewsFinder:
    """
    A class to fetch and filter news articles based on clickability predictions.
    """

    def __init__(self, prompt_text: str, llm: Any) -> None:
        """
        Initializes the class with a language model and a structured output pipeline for evaluating click probabilities.

        Parameters:
        ----------
        prompt_text : str
            The text template that will be used to generate the prompt for the LLM.
        llm : Any
            The language model (LLM) instance, expected to have a method `with_structured_output`.
        """
        try:
            self.model = llm.with_structured_output(WouldClickList)
            self.prompt = ChatPromptTemplate.from_template(prompt_text)
            self.chain = self.prompt | self.model
        except Exception as e:
            logger.error(f"Failed to initialize the LLM chain with the given prompt: {e}")
            raise

    @staticmethod
    def fetch_titles(sources: List[PageParser]) -> Generator[List[Dict[str, str]], None, None]:
        """
        Fetches article titles from a list of news sources.

        Parameters:
        ----------
        sources : List[PageParser]
            A list of PageParser instances representing different news sources.

        Yields:
        ----------
        List[Dict[str, str]]
            A list of dictionaries representing the titles and URLs fetched from each source.
        """
        for source in sources:
            try:
                yield source.get_article_titles()
            except Exception as e:
                logger.error(f"Error fetching titles from source '{source.name}': {e}")

    def remove_duplicate_links(self, links: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Removes duplicate links based on URLs.

        Parameters:
        ----------
        links : List[Dict[str, str]]
            A list of article links.

        Returns:
        ----------
        List[Dict[str, str]]
            A list of unique article links.
        """
        seen_urls = set()
        cleaned_links = []
        for link in links:
            url = link['url']
            if url not in seen_urls:
                seen_urls.add(url)
                cleaned_links.append(link)
        return cleaned_links

    def evaluate_clickability(self, link_titles: List[str]) -> List[bool]:
        """
        Evaluates the clickability of article titles using the language model.

        Parameters:
        ----------
        link_titles : List[str]
            A list of article titles.

        Returns:
        ----------
        List[bool]
            A list indicating whether each title is likely to be clicked.
        """
        keep_list: List[bool] = []
        batch_size: int = 10
        for i in trange(0, len(link_titles), batch_size, desc="Evaluating Clickability"):
            batch_titles = link_titles[i:i + batch_size]
            try:
                response = self.chain.invoke({"titles": batch_titles})
                keep_list.extend([r.would_click for r in response.would_click_list])
            except Exception as e:
                logger.error(f"AI model invocation failed for batch {i // batch_size}: {e}")
                keep_list.extend([False] * len(batch_titles))
        return keep_list

    def filter_titles(self, links: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Filters a list of article links by removing duplicates and determining which links are likely to be clicked.

        Parameters:
        ----------
        links : List[Dict[str, str]]
            A list of dictionaries, each representing an article link with keys 'title' and 'url'.

        Returns:
        ----------
        Tuple[List[Dict[str, str]], List[Dict[str, str]]]
            - A list of dictionaries for the links that are predicted to be clickable.
            - A list of dictionaries for the links that are predicted to be avoidable.
        """
        cleaned_links = self.remove_duplicate_links(links)
        link_titles = [link["title"] for link in cleaned_links]
        keep_list = self.evaluate_clickability(link_titles)
        clickable_links = list(compress(cleaned_links, keep_list))
        avoidable_links = list(compress(cleaned_links, [not k for k in keep_list]))
        return clickable_links, avoidable_links

    def find_news(self, sources: List[PageParser]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Processes a list of news sources to find clickable and avoidable links.

        Parameters:
        ----------
        sources : List[PageParser]
            A list of PageParser instances representing different news sources.

        Returns:
        ----------
        Tuple[List[Dict[str, str]], List[Dict[str, str]]]
            - A list of clickable links.
            - A list of avoidable links.
        """
        clickable_links: List[Dict[str, str]] = []
        avoidable_links: List[Dict[str, str]] = []

        for source_titles in self.fetch_titles(sources):
            try:
                clickable, avoidable = self.filter_titles(source_titles)
                clickable_links.extend(clickable)
                avoidable_links.extend(avoidable)
            except Exception as e:
                logger.error(f"Error processing source titles: {e}")

        return clickable_links, avoidable_links


class NewsRetriever:
    """
    A class to retrieve and add content to news articles.
    """

    @staticmethod
    def retrieve(news: List[Dict[str, Any]], sources: Dict[str, PageParser]) -> List[Dict[str, Any]]:
        """
        Retrieves and adds the content for each article in the news list by using the provided sources.

        Parameters:
        ----------
        news : List[Dict[str, Any]]
            A list of dictionaries representing news articles.
        sources : Dict[str, PageParser]
            A dictionary mapping source names to PageParser instances.

        Returns:
        ----------
        List[Dict[str, Any]]
            A list of dictionaries representing the news articles with content and timestamp.
        """
        news_with_content = []
        for article in tqdm(news, desc="Retrieving article content"):
            try:
                source_name = article["source"]
                source = sources.get(source_name)

                if "content" in article and article["content"]:
                    news_with_content.append(article)
                    continue

                if source:
                    article["content"] = source.get_article(article["url"])
                    article["image"] = source.get_article_image(article["url"])
                    print(article["image"])
                    article["timestamp"] = time.time()
                    news_with_content.append(article)
                else:
                    logger.warning(f"Source '{source_name}' not found for article with URL: {article['url']}")

            except KeyError as e:
                logger.error(f"Missing key {e} in article: {article}")
            except Exception as e:
                logger.error(f"Failed to retrieve content for article {article.get('url', 'unknown')}: {e}")


        return news_with_content


class Summary(BaseModel):
    summary: str = Field("The text of the summary")
    category: str = Field("The category of the article")


class Summarizer:
    """
    A class to summarize news articles and categorize them.
    """

    def __init__(self, prompt_text: str, llm: Any, categories: List[str] = NEWS_CATEGORIES):
        """
        Initializes the Summarizer with a language model and a prompt template.

        Parameters:
        ----------
        prompt_text : str
            The prompt template text used to guide the language model.
        llm : Any
            The language model instance that is capable of producing structured output.
        categories : List[str], optional
            A list of possible categories for articles.
        """
        self.model = llm.with_structured_output(Summary)
        self.prompt = ChatPromptTemplate.from_template(prompt_text)
        self.chain = self.prompt | self.model
        self.categories = categories

    def run(self, news: List[Dict[str, Any]], categories: List[str] = None) -> List[Dict[str, Any]]:
        """
        Summarizes articles and categorizes them using the language model.

        Parameters:
        ----------
        news : List[Dict[str, Any]]
            A list of news articles.
        categories : List[str], optional
            A list of categories to classify the articles.

        Returns:
        ----------
        List[Dict[str, Any]]
            A list of news articles with 'summary' and 'category' added.
        """
        news_with_summaries = []
        categories = categories or self.categories

        for article in tqdm(news, desc="Summarizing articles"):
            try:
                if "content" not in article or not article["content"]:
                    logger.warning(f"No content available for article: {article.get('url', 'unknown')}")
                    continue

                result = self.chain.invoke({
                    "article": article["content"],
                    "categories": categories
                })


                if not result or not hasattr(result, 'summary') or not hasattr(result, 'category'):
                    logger.error(f"Incomplete model output for article: {article.get('url', 'unknown')}")
                    continue

                article["summary"] = result.summary
                article["category"] = result.category
                news_with_summaries.append(article)

            except KeyError as e:
                logger.error(f"KeyError for article: {article.get('url', 'unknown')}. Missing key: {e}")
            except Exception as e:
                logger.error(f"Failed to summarize article with URL: {article.get('url', 'unknown')}. Error: {e}")

        return news_with_summaries


class TopicExtractor:
    """
    A class to extract topics from news articles.
    """

    def __init__(self, embedding_model: Any):
        self.embedding_model = embedding_model

    def extract_topics(self, news: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """
        Clusters news articles into topics based on their embeddings.

        Parameters:
        ----------
        news : List[Dict[str, Any]]
            A list of news articles.

        Returns:
        ----------
        Dict[int, List[Dict[str, Any]]]
            A dictionary where keys are cluster IDs and values are lists of news articles.
        """
        if not news:
            logger.warning("No news articles to process.")
            return {}

        embeddings = self.embed_news(news)
        clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.35, metric='cosine', linkage='average')
        labels = clustering_model.fit_predict(embeddings)

        clusters = {}
        for i, article in enumerate(news):
            cluster_id = int(labels[i])
            clusters.setdefault(cluster_id, []).append(article)

        return clusters

    def embed_news(self, news: List[Dict[str, Any]]) -> np.ndarray:
        """
        Embeds the news articles into vector representations using the provided model.

        Parameters:
        ----------
        news : List[Dict[str, Any]]
            A list of news articles.

        Returns:
        ----------
        np.ndarray
            A matrix of embeddings.
        """
        contents = [article["summary"] for article in news]
        return np.array(self.embedding_model.embed_documents(contents))


class Article(BaseModel):
    title: str = Field("The title of the article")
    content: str = Field("The content of the article")
    abstract: str = Field("The abstract of the article")


class Editor:
    """
    A class to create full articles from drafts.
    """

    def __init__(self, prompt_text: str, llm: Any, min_articles_per_topic: int = 2):
        """
        Initializes the Editor with a language model and a prompt template.

        Parameters:
        ----------
        prompt_text : str
            The prompt template text used to guide the language model.
        llm : Any
            The language model instance that is capable of producing structured output.
        """
        self.model = llm.with_structured_output(Article)
        self.prompt = ChatPromptTemplate.from_template(prompt_text)
        self.chain = self.prompt | self.model
        self.min_articles_per_topic = min_articles_per_topic

    def create_article(self, drafts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Creates a complete article based on provided drafts.

        Parameters:
        ----------
        drafts : List[Dict[str, Any]]
            A list of drafts.

        Returns:
        ----------
        Dict[str, Any]
            A dictionary representing the created article.
        """
        try:
            article = self.chain.invoke({"notes": drafts})
            return {
                "title": article.title,
                "abstract": article.abstract,
                "content": article.content,
            }
        except Exception as e:
            logger.error(f"Error while creating article: {e}")
            return {}

    def _create_drafts(self, topic_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Converts topic articles into draft format.

        Parameters:
        ----------
        topic_articles : List[Dict[str, Any]]
            A list of articles.

        Returns:
        ----------
        List[Dict[str, Any]]
            A list of draft dictionaries.
        """
        drafts = []
        for article in topic_articles:
            drafts.append({
                "title": article["title"],
                "content": article["content"],
                "source": article.get("source"),
                "url": article.get("url"),
                "image": article.get("image"),
            })
        return drafts

    def _extract_sources(self, drafts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts sources from drafts.

        Parameters:
        ----------
        drafts : List[Dict[str, Any]]
            A list of drafts.

        Returns:
        ----------
        List[Dict[str, Any]]
            A list of source dictionaries.
        """
        return [{"source": d.get("source"), "url": d.get("url"), "link": d.get("title"), "image": d.get("image")} for d in drafts]

    def create_articles(self, topic_articles: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Creates multiple articles from topic articles.

        Parameters:
        ----------
        topic_articles : Dict[int, List[Dict[str, Any]]]
            A dictionary of topic articles.
        max_articles : int
            Maximum number of articles to create.

        Returns:
        ----------
        List[Dict[str, Any]]
            A list of fully generated articles.
        """
        articles = []
        for topic in tqdm(topic_articles.values(), desc="Creating articles"):
            if len(topic) < self.min_articles_per_topic:
                continue

            drafts = self._create_drafts(topic)
            article = self.create_article(drafts)
            if article:
                article["sources"] = self._extract_sources(drafts)
                article["no_references"] = len(article["sources"])
                articles.append(article)
            else:
                logger.error("Article creation failed for topic.")
        return articles


def main():
    try:
        # Initialize the models
        model = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

        # Ensure directories exist

        os.makedirs("data", exist_ok=True)
        os.makedirs("prompts", exist_ok=True)

        # Read prompt texts
        try:
            with open("prompts/news.txt") as f:
                fetch_news_prompt_text = f.read()
            with open("prompts/summary.txt") as f:
                summary_prompt_text = f.read()
            with open("prompts/editor.txt") as f:
                editor_prompt_text = f.read()
        except FileNotFoundError as e:
            logger.error(f"Prompt file not found: {e}")
            return

        # Fetch and filter news articles
        logger.info("Fetching and filtering news...")
        news_finder = NewsFinder(fetch_news_prompt_text, model)
        news, spam = news_finder.find_news(news_sources)

        # Save news and spam to JSON files
        with open("data/news.json", "w") as f:
            json.dump(news, f, indent=4)
        with open("data/reject_news.json", "w") as f:
            json.dump(spam, f, indent=4)
        logger.info("News fetching and filtering completed. News and spam saved.")

        # Retrieve content for the filtered news articles
        logger.info("Retrieving content for news articles...")
        news = NewsRetriever.retrieve(news, news_mapping)

        # Save news with content to a JSON file
        with open("data/news_with_content.json", "w") as f:
            json.dump(news, f, indent=4)
        logger.info("Content retrieval completed. News with content saved.")

        # Summarize the news articles
        logger.info("Summarizing news articles...")
        summarizer = Summarizer(summary_prompt_text, model)
        news = summarizer.run(news)

        # Save news with summaries to a JSON file
        with open("data/news_with_summaries.json", "w") as f:
            json.dump(news, f, indent=4)
        logger.info("Summarization completed. News with summaries saved.")

        # Extract topics from news articles
        logger.info("Extracting topics...")
        extractor = TopicExtractor(embeddings_model)
        clusters = extractor.extract_topics(news)


        # Save clusters to a JSON file
        with open("data/clusters.json", "w") as f:
            json.dump(clusters, f, indent=4)
        logger.info("Topic extraction completed. Clusters saved.")

        # Create articles from clusters
        logger.info("Creating articles...")
        editor = Editor(editor_prompt_text, model)
        articles = editor.create_articles(clusters)

        # Save articles to a JSON file
        with open("data/articles.json", "w") as f:
            json.dump(articles, f, indent=4)
        logger.info("Article creation completed. Articles saved.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()