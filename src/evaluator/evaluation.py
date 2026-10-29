import json
from typing import Dict, Tuple, List
from src.models.models import (AnsweredQuestion,
                               MinimalSearchResults,)


def loading_files(
    student_path: str,
    sys_path: str
) -> Tuple[List[MinimalSearchResults], List[AnsweredQuestion]]:
    """
    this function loads student search results and the reference
    dataset from JSON files and validates their contents
    using Pydantic models

    args:
        student_path: str = path to the student search results
        sys_path: str = path to the reference answered dataset

    return:
        tuple containing a list of MinimalSearchResults objects
        and a list of AnsweredQuestion objects
    """
    with open(student_path, "r") as file:
        student = json.load(file)

    min_searches = []
    for quest in student["search_results"]:
        min_searches.append(
            MinimalSearchResults(**quest)
        )
    with open(sys_path, "r") as f:
        system = json.load(f)

    ans_quest = []
    for quest in system["rag_questions"]:
        ans_quest.append(
            AnsweredQuestion(**quest)
        )
    return min_searches, ans_quest


def intersection_over_union(student_search_results_path: str,
                            dataset_path: str) -> List[str | Dict[str, float]]:
    """
    this function evaluates retrieved sources against reference
    sources using intersection over union (IoU) with a threshold
    of 0.05 and calculates recall at ranks 1, 3, 5 and 10

    args:
        student_search_results_path: str = path to student results
        dataset_path: str = path to the reference answered dataset

    return:
        list containing recall percentages and a dictionary
        of normalized recall@k scores
    """
    if not isinstance(student_search_results_path, str):
        raise TypeError("[ERROR]: Student search result path must be str "
                        "and available!")
    if not isinstance(dataset_path, str):
        raise TypeError("[ERROR]: dataset path must be str "
                        "and available!")
    student, system = loading_files(student_search_results_path,
                                    dataset_path)

    ids: Dict[str, AnsweredQuestion] = {}
    recallk: Dict[str, int | float] = {
        "recall@1": 0,
        "recall@3": 0,
        "recall@5": 0,
        "recall@10": 0
    }

    for sys in system:
        if sys.question_id in ids.keys():
            continue
        ids[sys.question_id] = sys

    seen = set()
    for search in student:
        if search.question_id in seen:
            continue
        seen.add(search.question_id)

        if search.question_id in ids.keys():
            obj = ids[search.question_id].sources[0]
            for index, source in enumerate(search.retrieved_sources, 1):
                if source.file_path == obj.file_path:
                    a_start = source.first_character_index
                    b_start = obj.first_character_index
                    a_end = source.last_character_index
                    b_end = obj.last_character_index

                    inter = max(
                        0, min(a_end, b_end) - max(a_start, b_start)
                    )
                    union = (a_end - a_start) + (b_end - b_start) - inter
                    try:
                        IoU = inter / union
                    except ZeroDivisionError:
                        IoU = 0

                    if IoU >= 0.05:
                        if index == 1:
                            recallk["recall@1"] += 1
                            recallk["recall@3"] += 1
                            recallk["recall@5"] += 1
                            recallk["recall@10"] += 1
                            break
                        elif index <= 3:
                            recallk["recall@3"] += 1
                            recallk["recall@5"] += 1
                            recallk["recall@10"] += 1
                            break
                        elif index <= 5:
                            recallk["recall@5"] += 1
                            recallk["recall@10"] += 1
                            break
                        elif index <= 10:
                            recallk["recall@10"] += 1
                            break
    results: List[str | Dict[str, float]] = []
    results.append("Results are:")
    for k, v in recallk.items():
        results.append(f"{v}%")
        recallk[k] = v / 100

    results.append(recallk)
    return results
