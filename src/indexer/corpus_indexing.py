import os
from tqdm import tqdm
from math import log as ln
from typing import List, Dict, Tuple
from src.models.models import MinimalSource
from src.indexer.tokenizer import tokenizer
from src.indexer.chunker_py import python_chunker
from src.indexer.chunker_md import markdown_chunker


# ======== Gathering all the .py and .md files ===========
def walk_corpus(dir_path: str) -> List[str]:
    """
    This function walks through all the files given in a directory
    using the os.walk method

    args:
        dir_path: str = directory path string

    return:
        a list of path strings, containing all python, markdown and text files
    """
    lst = []
    path = os.walk(dir_path)

    for root, dirs, files in path:
        for file in files:
            if file.endswith(".py") or file.endswith((".md", ".txt")):
                lst.append(os.path.join(root, file))

    if not lst:
        raise ValueError("[ERROR]: No files/directory Detected")
    return lst


# ========== Chunking all the .py and .md files ============
def chunk_corpus(dir_path: str, max_size: int) -> List[MinimalSource]:
    """
    this functions gets all the files from the directories and uses
    the python and markdown chunkers

    args:
        dir_path: str = directory path
        max_size: int = max chunking size, no more than 2000 and no less than 1

    return:
        list of MinimalSource objects
    """
    files = walk_corpus(dir_path)
    if not files:
        raise ZeroDivisionError("[ERROR]: No file is found!")
    chunks = []

    for file in tqdm(files, desc="Chunking"):
        if file.endswith(".py"):
            data = python_chunker(file, max_size)
            chunks.extend(data)
        if file.endswith((".md", ".txt")):
            data = markdown_chunker(file, max_size)
            chunks.extend(data)

    return chunks


# ========= Term Frequency (TF) =========
def inverted_indexing(chunks: List[MinimalSource]) -> Dict:
    """
    this function builds an inverted index by tokenizing each chunk
    and storing the chunk index and term frequency for every unique token

    args:
        chunks: List[MinimalSource] = list of source chunks to index

    return:
        dictionary mapping tokens to lists of (chunk_index, frequency)
    """
    dct: Dict[str, List[Tuple[int, int]]] = {}

    for index, chunk in tqdm(enumerate(chunks), desc="TF",
                             total=len(chunks)):
        seen = set()
        tokens = tokenizer(chunk.content)
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            count = tokens.count(token)
            if token not in dct.keys():
                dct[token] = [(index, count)]
            else:
                dct[token].append((index, count))

    return dct


# ======== inverse document frequency (IDF) ========
def inverse_df(chunks: List, tokens: Dict) -> Dict:
    """
    this function calculates the inverse document frequency (IDF)
    of each token using the BM25 IDF formula

    args:
        chunks: List = list of chunks used to calculate total documents
        tokens: Dict = inverted index containing tokens and their frequencies

    return:
        dictionary mapping each token to its IDF score
    """
    n = len(chunks)
    token_idf = {}

    for key in tqdm(tokens.keys(), desc="IDF"):
        df = len(tokens[key])
        idf = ln(1 + ((n - df + 0.5) / (df + 0.5)))
        token_idf[key] = idf

    return token_idf


# ========== Document Length (DL) ==========
def document_length(chunks: List[MinimalSource]) -> Dict[str, int]:
    """
    this function calculates the length of each chunk by counting
    the number of tokens it contains

    args:
        chunks: List[MinimalSource] = list of source chunks

    return:
        dictionary mapping each chunk index to its token count
    """
    dl = {}

    total = len(chunks)
    for index, chunk in tqdm(enumerate(chunks), desc="Tokenizing",
                             total=total):
        tokens = tokenizer(chunk.content)
        dl[f'Chunk[{index}]'] = len(tokens)

    return dl


# =========== Average Document Length (avgdl) ==========
def avg_doc_len(dls: Dict) -> float:
    """
    this function calculates the length of each chunk by counting
    the number of tokens it contains

    args:
        chunks: List[MinimalSource] = list of source chunks

    return:
        dictionary mapping each chunk index to its token count
    """
    result = sum([v for v in dls.values()]) / len([k for k in dls.keys()])
    assert isinstance(result, float)
    return result
