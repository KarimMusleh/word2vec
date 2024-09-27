import wikipediaapi

# Initialize the Wikipedia API
wiki = wikipediaapi.Wikipedia("YourProjectName/1.0 (your-email@example.com)",'en')

# Function to retrieve clean text and links from a Wikipedia page
def get_page_content_and_links(title):
    page = wiki.page(title)
    if not page.exists():
        return None, None
    text = page.text  # Get the clean text
    links = list(page.links.keys())  # Get the links from the page
    return text, links

# Create a dataset from a Wikipedia page, given a certain depth
def create_wikipedia_dataset(start_title, depth=1, max_width=100):
    pages_scraped = set()  # To avoid visiting the same page twice
    dataset = []

    # Initialize with the main page
    pages_to_scrape = [start_title]

    for d in range(-1,depth):
        next_pages_to_scrape = []
        i = 0
        while len(dataset) < max_width and i < len(pages_to_scrape):
            page_title = pages_to_scrape[i]
            i += 1
            if page_title not in pages_scraped:
                clean_text, links = get_page_content_and_links(page_title)
                if clean_text:
                    dataset.append(clean_text)
                    pages_scraped.add(page_title)
                if links:
                    next_pages_to_scrape.extend(links)
        # Prepare for the next depth layer
        pages_to_scrape = next_pages_to_scrape

    return dataset

# Usage example
main_title = "philosophy"  # This is the title of the Wikipedia page
dataset = create_wikipedia_dataset(main_title, depth=1, max_width=200)  # 1-level deep

# Save datasets to files if needed
with open('phil_dataset.txt', 'w') as f:
    curr = f.write('\n__WIKI__'.join(dataset))


