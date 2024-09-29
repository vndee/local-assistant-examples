from collections import deque
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from typing import Literal, List, Dict
from playwright.sync_api import sync_playwright, Page, Error
from langchain_community.chat_models import ChatOllama, ChatOpenAI, ChatAnthropic

TLLMProvider = Literal["ollama", "openai", "anthropic"]


class ScrapingAssistant:
    def __init__(
        self,
        root_url: str = "https://blog.duy.dev",
        max_pages: int = 100,
        max_depth: int = 2,
        llm_provider: TLLMProvider = "openai",
        llm_model: str = "gpt-4o-mini",
        verbose: bool = True,
    ):
        """
        Intelligent assistant for scraping websites using BFS.
        :param root_url: The root URL to start scraping from.
        :param max_pages: The maximum number of pages to scrape.
        :param max_depth: The maximum depth of the scraping.
        :param llm_provider: The LLM provider to use for generating responses.
        :param llm_model: The LLM model to use for generating responses.
        :param verbose: Whether to log detailed information.
        """
        self.root_url = self.normalize_url(root_url)
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.verbose = verbose
        self.pages_scraped: List[str] = []
        self.visited_links: Dict[str, str] = {}
        self.bfs_tree: Dict[str, List[str]] = {}
        self.llm = self.init_llm(llm_provider, llm_model)

    @staticmethod
    def init_llm(provider: TLLMProvider, model: str):
        """Initialize the appropriate LLM based on the provider."""
        if provider == "ollama":
            return ChatOllama(model=model)
        elif provider == "openai":
            return ChatOpenAI(model=model)
        elif provider == "anthropic":
            return ChatAnthropic(model=model)
        raise ValueError("Invalid LLM provider")

    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize URLs to treat `https://blog.duy.dev` and `https://blog.duy.dev/` as the same."""
        parsed_url = urlparse(url)
        return f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path.rstrip('/')}"

    @staticmethod
    def extract_content(page: Page) -> str:
        """Extracts the main text content from the page."""
        html_content = page.content()
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script, style, and navigation elements
        for element in soup(['script', 'style', 'nav', 'footer', 'noscript']):
            element.decompose()

        if soup.find('article'):
            main_content = soup.find('article').get_text(separator=' ', strip=True)
        elif soup.find('main'):
            main_content = soup.find('main').get_text(separator=' ', strip=True)
        else:
            # Use heuristic to find the largest div
            divs = soup.find_all('div')
            if divs:
                main_content = max(divs, key=lambda d: len(d.get_text())).get_text(separator=' ', strip=True)
            else:
                # Fallback to extracting all text
                main_content = soup.get_text(separator=' ', strip=True)

        return main_content

    @staticmethod
    def extract_links(page: Page, current_url: str) -> List[str]:
        """Extracts valid links from a page."""
        links = page.query_selector_all("a")
        return [
            urljoin(current_url, link.get_attribute("href"))
            for link in links
            if link.get_attribute("href")
            and not link.get_attribute("href").startswith("#")
        ]

    def is_same_domain(self, url: str) -> bool:
        """Checks if a URL belongs to the same domain as the root URL."""
        return urlparse(url).netloc == urlparse(self.root_url).netloc

    def run(self):
        """Runs the scraping assistant using BFS."""
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()

            self.bfs(page, self.root_url)
            browser.close()

    def bfs(self, page: Page, start_url: str):
        """Performs breadth-first search to scrape the website."""
        queue = deque([{"url": start_url, "depth": 0}])
        parent = {start_url: None}
        visited = set()

        while queue and len(self.pages_scraped) < self.max_pages:
            current = queue.popleft()
            current_url, current_depth = current["url"], current["depth"]

            if current_url in visited or current_depth > self.max_depth:
                continue

            visited.add(current_url)
            normalized_url = self.normalize_url(current_url)

            try:
                page.goto(normalized_url)
                content = self.extract_content(page)
                self.pages_scraped.append(content)
                self.visited_links[normalized_url] = normalized_url
                if self.verbose:
                    print(f"Scraped {normalized_url} at depth {current_depth}")

                children_links = self.extract_links(page, normalized_url)
                for child_url in filter(self.is_same_domain, children_links):
                    normalized_child_url = self.normalize_url(child_url)
                    if normalized_child_url not in parent:
                        parent[normalized_child_url] = current_url
                        queue.append(
                            {"url": normalized_child_url, "depth": current_depth + 1}
                        )
                        self.bfs_tree.setdefault(current_url, []).append(
                            normalized_child_url
                        )
            except Error as e:
                if self.verbose:
                    print(f"Failed to load {normalized_url}: {e}")

    def print_bfs_tree(self, is_scraped_only: bool = True):
        """Prints the BFS tree in a flat, depth-first style, similar to the `tree` command."""

        def print_tree(node: str, indent: str = "", visited=None):
            if visited is None:
                visited = set()
            if node in visited:
                return
            visited.add(node)

            children = self.bfs_tree.get(node, [])

            for idx, child in enumerate(children):
                if not is_scraped_only or (
                    is_scraped_only and child in self.visited_links
                ):
                    connector = "└── " if idx == len(children) - 1 else "├── "
                    print(f"{indent}{connector}{child}")

                    new_indent = indent + (
                        "    " if idx == len(children) - 1 else "│   "
                    )
                    print_tree(child, new_indent, visited)

        print(self.root_url)
        print_tree(self.root_url)


if __name__ == "__main__":
    assistant = ScrapingAssistant(
        root_url="https://substack.com/home/post/p-149271488?source=queue",
        max_depth=3,
        max_pages=1,
    )
    assistant.run()
    assistant.print_bfs_tree(is_scraped_only=True)
    print(assistant.pages_scraped[0])
