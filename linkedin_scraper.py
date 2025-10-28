# linkedin_scraper.py
import requests
from bs4 import BeautifulSoup

def scrape_linkedin_jobs(query, location="India", limit=5):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    url = f"https://www.linkedin.com/jobs/search?keywords={query}&location={location}"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    jobs = []
    for card in soup.select(".base-card__full-link")[:limit]:
        title = card.text.strip()
        link = card["href"]
        company_el = card.find_previous("h4", class_="base-search-card__subtitle")
        company = company_el.text.strip() if company_el else "Unknown"
        jobs.append({
            "title": title,
            "company": company,
            "link": link
        })
    return jobs
