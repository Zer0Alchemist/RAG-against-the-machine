import os
import json
from tqdm import tqdm
from src.models.models import (MinimalAnswer,
                               StudentSearchResultsAndAnswer)
from src.retriever.bm25 import retrieve_top_k
from src.augmenter.augmentation import system_prompt
from src.generator.loading import LoadingLLM

p = "data/output/search_results/UnansweredQuestions/dataset_docs_results.json"


def generate_output(query: str, llm: LoadingLLM, k: int) -> str:
    """
    this function retrieves relevant sources, builds an augmented
    prompt, generates an answer using the language model
    and decodes the generated tokens into readable text

    args:
        query: str = user's question
        llm: LoadingLLM = initialized language model
        k: int = number of source chunks used for retrieval

    return:
        string containing the generated answer
    """
    sys_prompt = system_prompt(query, k)
    message = [
        {"role": "user", "content": query},
        {"role": "system", "content": sys_prompt}
    ]

    new_prompt = llm.build_template(message)
    encoded_lst = llm.encode(new_prompt)
    generated_lst = llm.generate(encoded_lst, 500)

    generated_index = generated_lst[0][len(encoded_lst[0]):]
    output = llm.decode(generated_index)

    return output


def answer_dataset(
    student_search_results_path: str = p,
    k: int = 10,
    save_directory: str = "data/output/search_results/UnansweredQuestions",
) -> list[str]:
    """
    this function loads a dataset of search results, generates
    answers for each question using Qwen and saves the answers
    with their retrieved sources into a JSON file

    args:
        search_path: str = path to the student search results JSON
        k: int = number of retrieved source chunks
        save_directory: str = directory for saving generated answers

    return:
        list of strings describing the processing and output status
    """
    if not isinstance(student_search_results_path, str):
        raise TypeError("[ERROR]: search_path must be str and available!")
    if not isinstance(k, int):
        raise TypeError("[ERROR]: k must be int!")
    if not isinstance(save_directory, str):
        raise TypeError("[ERROR]: save_directory must be str and available!")

    with open(student_search_results_path, "r") as file:
        questions = json.load(file)

    min_answers = []
    llm = LoadingLLM("Qwen/Qwen3-0.6B")
    output = []
    for quest in tqdm(questions["search_results"],
                      desc="Generating"):
        top_k = retrieve_top_k(quest["question"], k)
        answered_q = generate_output(quest["question"], llm, k)

        min_answers.append(
            MinimalAnswer(question_id=quest["question_id"],
                          question=quest["question"],
                          retrieved_sources=top_k,
                          answer=answered_q)
        )
    output.append(f"Loaded {len(questions['search_results'])} questions ... "
                  f"Processed {len(questions['search_results'])} questions")
    answer_res = StudentSearchResultsAndAnswer(
        search_results=min_answers,
        k=k
    )

    dir_path = save_directory
    os.makedirs(dir_path, exist_ok=True)
    if "docs" in os.path.basename(student_search_results_path).split("_"):
        file_path = os.path.join(dir_path, "answer_docs_results.json")
    elif "code" in os.path.basename(student_search_results_path).split("_"):
        file_path = os.path.join(dir_path, "answer_code_results.json")
    else:
        file_path = os.path.join(dir_path, "dataset_results.json")

    with open(file_path, "w") as file:
        file.write(answer_res.model_dump_json())

    output.append("Saved student_search_results_and_answer to "
                  f"{file_path}")

    return output


def answer(query: str | None = None, k: int = 10,
           model_name: str = "Qwen/Qwen3-0.6B") -> str:
    """
    this function answers a single query by loading the specified
    language model and generating a response using retrieved context

    args:
        query: str | None = user's question
        k: int = number of source chunks used for retrieval
        model_name: str = name or path of the pretrained model

    return:
        string containing the generated answer
    """
    if k == 0:
        raise ValueError("[ERROR]: k must be larger than 0")
    if not query:
        raise ValueError("[ERROR]: no query detected!")
    llm = LoadingLLM(model_name)

    return generate_output(query, llm, k)
