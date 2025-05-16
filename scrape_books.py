import requests

year = 1800
language = "en"

url = f"https://gutendex.com/books?author_year_start={year}&languages={language}"


with open("corpus.txt", "w") as file:
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        books = data.get("results", [])

        for index, book in enumerate(books):
            title = book.get("title")
            author = book.get("authors", [{}])[0].get("name", "Unknown Author")
            print(f"{index + 1}. Title: {title}, Author: {author}")

            # Download the book into a text file
            book_url = book.get("formats", {}).get("text/plain; charset=us-ascii")
            if book_url:
                book_response = requests.get(book_url)
                if book_response.status_code == 200:
                    file.write(f"Title: {title}\n")
                    file.write(f"Author: {author}\n")
                    file.write(book_response.text)
                    file.write("\n\n")
                else:
                    print(f"Failed to download book: {title}")
        print(f"Downloaded {len(books)} books into 'corpus.txt'")
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")
