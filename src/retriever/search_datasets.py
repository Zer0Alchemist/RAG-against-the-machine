import json
import os
from tqdm import tqdm
from src.models.models import (StudentSearchResults,
                               MinimalSearchResults,
                               RagDataset)
from src.retriever.bm25 import retrieve_top_k

def_path = "data/datasets/UnansweredQuestions/dataset_docs_public.json"


def search_dataset(
    dataset_path: str = def_path,
    k: int = 10,
    save_directory: str = "data/output/search_results/UnansweredQuestions"
) -> str:
    """
    this function loads a dataset of unanswered questions,
    retrieves the top k relevant source chunks for each question
    using BM25, and saves the search results into a JSON file

    args:
        dataset_path: str = path to the questions dataset JSON file
        k: int = number of source chunks to retrieve, default is 10
        save_directory: str = directory where results will be saved

    return:
        string confirming the location of the saved results
    """
    if not isinstance(dataset_path, str) or not dataset_path.endswith(".json"):
        raise TypeError("[ERROR]: dataset_path must be a JSON file!")
    if not isinstance(k, int):
        raise TypeError("[ERROR]: k must be an int")
    if not isinstance(save_directory, str):
        raise TypeError("[ERROR]: save_directory must be a string with "
                        "available path!")

    with open(dataset_path, "r") as file:
        questions = RagDataset(**json.load(file))

    min_answers = []
    for quest in tqdm(questions.rag_questions,
                      desc="Searching"):
        top_k = retrieve_top_k(quest.question, k)
        min_answers.append(
            MinimalSearchResults(
                question_id=quest.question_id,
                question=quest.question,
                retrieved_sources=top_k,
            )
        )
    search_res = StudentSearchResults(search_results=min_answers, k=k)

    dir_path = save_directory
    os.makedirs(dir_path, exist_ok=True)
    if "docs" in os.path.basename(dataset_path).split("_"):
        file_path = os.path.join(dir_path, "dataset_docs_results.json")
    elif "code" in os.path.basename(dataset_path).split("_"):
        file_path = os.path.join(dir_path, "dataset_code_results.json")
    else:
        file_path = os.path.join(dir_path, "dataset_results.json")

    with open(file_path, "w") as file:
        file.write(search_res.model_dump_json())

    return f"Saved student_search_restuls to {file_path}"
