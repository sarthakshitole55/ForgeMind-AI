from app.services.search_service import TavilySearch
search = TavilySearch()

result = search.search("Latest Python release")
for i ,result in enumerate(result,start=1):
    print("=" * 80)
    print(f"Result {i}")
    print("Title :", result["title"])
    print("URL   :", result["url"])
    print()
    print(result["content"])