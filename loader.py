def load_in_text_corpus(file_path: str) -> str:
    """Load a text corpus from a file.

    Parameters
    ----------
    file_path : str
        The path to the text file.

    Returns
    -------
    str
        The content of the text file.
    """
    with open(file_path, "r") as file:
        return file.read()
