from typing import List
from src.utils.file_utils import read_file
from src.models.models import MinimalSource
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


def markdown_chunker(file_path: str, max_size: int) -> List[MinimalSource]:
    """
    This function chunks all the markdown/text (.md/.txt)files given
    by using langchain text splitter class with from language method
    and create documents method

    args:
        file_path: str = markdown file path
        max_size = max_chunk_size no more than 2000 and no less than 1

    returns:
        A list of MinimalSource objects, for indexing use
    """
    if max_size <= 0:
        raise ValueError("[ERROR] max_size is less or equal to 0")

    content = read_file(file_path)
    if not content:
        return []

    contents = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=max_size,
        chunk_overlap=min(100, max_size // 5),
        add_start_index=True,
        strip_whitespace=False
    )

    all_content = contents.create_documents([content])
    chunks = []

    for cont in all_content:
        lst_char = cont.metadata["start_index"] + len(cont.page_content)
        chunk = MinimalSource(
            file_path=file_path,
            first_character_index=cont.metadata["start_index"],
            last_character_index=lst_char,
            content=cont.page_content
        )
        chunks.append(chunk)

    return chunks
