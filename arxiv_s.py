import arxiv
import json


def fetch_abstracts(query, max_results=50):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
    )

    abstracts = []
    for result in search.get():
        abstracts.append({'abs': result.summary, 'title': result.title})

    return abstracts

if __name__ == "__main__":
    abs_list = []
    for year in range(2022, 2023):
        query = f"MM {year}"
        abstracts = fetch_abstracts(query)
        abs_list.extend(abstracts)

    with open(f'arXiv_human.json', 'w') as file:
        json.dump(abs_list, file, indent=4)