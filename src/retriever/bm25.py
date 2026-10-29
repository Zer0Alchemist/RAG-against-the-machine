import pickle
from typing import List, Tuple, Dict
from src.indexer.tokenizer import tokenizer
from src.models.models import MinimalSource


# =============== BM25 full calucation ========================
def bm_25(query: str, index: Dict) -> List[Tuple[int, float]]:
    """
    this function calculates BM25 relevance scores for a query
    using the inverted index, term frequencies, IDF scores,
    document lengths and average document length
    then sorts the chunks by their scores in descending order

    args:
        query: str = user's search query
        index: Dict = dictionary containing the stored BM25 index

    return:
        list of tuples containing (chunk_index, bm25_score)
        sorted from highest to lowest score
    """
    bm_scores: Dict[int, float] = {}
    q_tokens = tokenizer(query)

    for token in q_tokens:
        if token in index["inverted_index"].keys():
            tok_tf = index["inverted_index"][token]
            for tok in tok_tf:
                first = index["idf"][token] * (tok[1] * (1.5 + 1))
                lst = 0.75 * index["dl"][f"Chunk[{tok[0]}]"] / index["avgdl"]
                last = (tok[1] + (1.5 * (1 - 0.75 + (lst))))
                bm25 = first / last
                if tok[0] in bm_scores.keys():
                    bm_scores[tok[0]] += bm25
                else:
                    bm_scores[tok[0]] = bm25
        else:
            continue

    sorted_scores = sorted(bm_scores.items(), key=lambda item: item[1],
                           reverse=True)

    return sorted_scores


# ======= top k retrieval ======
def retrieve_top_k(query: str, k: int = 10) -> List[MinimalSource]:
    """
    this function loads the serialized BM25 index, calculates
    the relevance scores for a query and retrieves the top k
    most relevant source chunks

    args:
        query: str = user's search query
        k: int = number of source chunks to retrieve, default is 10

    return:
        list of MinimalSource objects ranked by relevance
    """
    if k <= 0:
        raise ValueError("[ERROR]: k must be greater than 0")
    with open("data/processed/indexing.pkl", "rb") as file:
        index = pickle.load(file)
    scores = bm_25(query, index)
    if k > len(scores):
        raise ValueError("[ERROR]: k is larger than the length of the chunks!")

    chunks = []
    for n in range(k):
        chunks.append(index["chunks"][scores[n][0]])

    return chunks


def search(query: str | None = None, k: int = 5) -> List[str]:
    """
    this function searches for the top k relevant source chunks
    and formats their file paths and character ranges for display

    args:
        query: str | None = user's search query
        k: int = number of results to return, default is 5

    return:
        list of strings containing source file paths
        and their starting and ending character indices
    """
    if not query or not isinstance(query, str):
        raise ValueError("[ERROR]: No query detected!")
    top_k = retrieve_top_k(query, k)
    lst = []
    for path in top_k:
        start = path.first_character_index
        end = path.last_character_index
        lst.append(f"{path.file_path} [{start}:{end}]")

    return lst
