import pickle
import os
from src.indexer.corpus_indexing import chunk_corpus
from src.indexer.corpus_indexing import inverted_indexing
from src.indexer.corpus_indexing import inverse_df
from src.indexer.corpus_indexing import document_length
from src.indexer.corpus_indexing import avg_doc_len


def index(dir_path: str = "data/raw/vllm-0.10.1",
          max_chunk_size: int = 2000) -> str:
    """
    this function indexes the source corpus by chunking files,
    calculating term frequencies, IDF scores and document lengths,
    then serializes the final BM25 index using pickle

    args:
        dir_path: str = directory containing the source files
        max_chunk_size: int = maximum chunk size, between 1 and 2000

    return:
        string confirming where the index was saved
    """
    if not isinstance(dir_path, str):
        raise TypeError("[ERROR]: dir_path must be an available directory!")
    if max_chunk_size <= 0:
        raise ValueError("[ERROR]: max_chunk_size is less or equal to 0!")
    chunks = chunk_corpus(dir_path, max_chunk_size)
    term_f = inverted_indexing(chunks)
    idf = inverse_df(chunks, term_f)
    doc_length = document_length(chunks)
    average_dl = avg_doc_len(doc_length)

    final_index = {
        "chunks": chunks,
        "inverted_index": term_f,
        "idf": idf,
        "dl": doc_length,
        "avgdl": average_dl,
        "n_chunks": len(chunks)
    }

    dir_path = "data/processed"
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, "indexing.pkl")

    with open(file_path, "wb") as file:
        pickle.dump(final_index, file)

    return f"Ingestion complete! Indices saved under {file_path}"
