import json
import random

import html

article_row = """
 <div class="article-row">
    <div class="thumbnail">
        <img src="{img_url}" alt="{article_title}">
    </div>
    <div class="article-content">
        <h2 class="article-title">{article_title}</h2>
        <p class="short-description">{article_summary}</p>

        <!-- Bias Indicator -->
        <div class="fairness-info">
            <!-- Number of References -->
            <span class="reference-count">The article is based on {ref_no} references</span>
        </div>

        <a href="javascript:void(0);" class="read-more-link" id="link{no}"
            onclick="toggleArticle('article{no}', 'link{no}')">Read More</a>
        <div id="article{no}" class="full-article">
            <!-- Full Article Section -->
            <div class="full-article-section">
                <p>{article_content}</p>
            </div>
            <!-- Sources Section -->
            <div class="sources">
                <h3>References</h3>
                {sources}
            </div>
            <hr>
            <a href="javascript:void(0);" class="read-less-link" id="link{no}-bottom"
                onclick="toggleArticle('article{no}', 'link{no}')">Read Less</a>
        </div>
    </div>
</div>
"""


def load_articles(filename: str) -> list:
    """
    Loads articles from a JSON file.

    Parameters:
    ----------
    filename : str
        The path to the JSON file containing the articles.

    Returns:
    ----------
    List[Dict]
        A list of articles loaded from the JSON file.
    """
    with open(filename) as f:
        return json.load(f)


def sort_articles_by_references(articles: list) -> list:
    """
    Sorts articles based on the number of references in descending order.

    Parameters:
    ----------
    articles : List[Dict]
        A list of articles to be sorted.

    Returns:
    ----------
    List[Dict]
        A sorted list of articles.
    """
    return sorted(articles, key=lambda k: k.get('no_references', 0), reverse=True)


def select_thumbnail(sources: list) -> str:
    return random.choice([s["image"] for s in sources if s["image"]])


def generate_html_articles(articles: list) -> list:
    """
    Generates the HTML for the list of articles.

    Parameters:
    ----------
    articles : List[Dict]
        A list of articles to be formatted into HTML.

    Returns:
    ----------
    List[str]
        A list of formatted HTML strings for each article.
    """
    html_articles = []
    for no, article in enumerate(articles):
        # Handle missing fields and escape special characters
        title = html.escape(article.get("title", "Untitled"))
        abstract = html.escape(article.get("abstract", "No summary available."))
        content = html.escape(article.get("content", "No content available.")).replace("\n", "</br>")
        ref_no = article.get("no_references", 0)

        # Construct the sources section
        sources = ""
        for i, source in enumerate(article.get("sources", [])):
            source_link = html.escape(source.get("link", "Source").strip())
            source_url = html.escape(source.get("url", "#"))

            sources += f"<a href='{source_url}' target='_blank'>{i + 1}. {source_link.capitalize()[:100]}... [{source.get('source').replace('_', '.')}]</a><br>\n"

        thumbnail = select_thumbnail(article.get("sources", []))

        # Use a dynamic score based on the number of references (as an example)
        score = min(100, max(0, ref_no * 10))  # For example, a rough fairness score based on the number of references

        # Format the article row using the template
        unbiased_article = article_row.format(
            article_title=title,
            article_summary=abstract,
            no=no,
            article_content=content,
            sources=sources,
            ref_no=ref_no,
            score=score,
            img_url=thumbnail
        )

        # Add the formatted article to the HTML list
        html_articles.append(unbiased_article)

    return html_articles


def replace_template_placeholder(html_template: str, articles_section: str) -> str:
    """
    Replaces the placeholder in the HTML template with the articles section.

    Parameters:
    ----------
    html_template : str
        The HTML template as a string.

    articles_section : str
        The generated HTML for all the articles.

    Returns:
    ----------
    str
        The final HTML with the articles section inserted.
    """
    return html_template.replace("{articles_section}", articles_section)


def save_html_file(output_filename: str, html_content: str):
    """
    Saves the generated HTML content to a file.

    Parameters:
    ----------
    output_filename : str
        The file to save the HTML content.

    html_content : str
        The generated HTML content to be saved.
    """
    with open(output_filename, "w") as f:
        f.write(html_content)


def main():
    """
    Main function that generates an HTML page from the articles data.
    """
    # Step 1: Load articles from JSON
    articles = load_articles("data/articles.json")

    # Step 2: Sort articles by the number of references
    articles = sort_articles_by_references(articles)

    # Step 3: Generate HTML for the articles
    html_articles = generate_html_articles(articles)
    articles_section = "\n\n\n".join(html_articles)

    # Step 4: Load the HTML template
    with open("html/template.html") as f:
        html_template = f.read()

    # Step 5: Replace the placeholder with generated articles
    final_html = replace_template_placeholder(html_template, articles_section)

    # Step 6: Save the generated HTML to a file
    save_html_file("index.html", final_html)

    print("HTML page generated successfully.")


if __name__ == "__main__":
    main()
