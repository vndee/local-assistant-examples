import asyncio
from crawl4ai import AsyncWebCrawler


async def main():
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(
            url="https://vnexpress.net/israel-ha-sat-thu-linh-hezbollah-nhu-the-nao-4798374.html"
        )
        print(result.markdown)


if __name__ == "__main__":
    asyncio.run(main())
