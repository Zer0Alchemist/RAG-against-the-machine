from src.retriever.bm25 import retrieve_top_k


def system_prompt(query: str, k: int = 10) -> str:
    """
    this function retrieves the top k relevant source chunks
    and builds an augmented prompt containing instructions,
    source file paths, source content and the user's question

    args:
        query: str = user's question
        k: int = number of source chunks to retrieve, default is 10

    return:
        string containing the formatted prompt with retrieved context
    """
    data = retrieve_top_k(query, k)
    sys_prompt = [
        "system\n",
        "technical assistant answering questions using retrieved sources. ",
        "Your ONLY task is to generate ONE answer to the user's question ",
        "based on the retrived sources. "
        "If the user's question does not have any relation to the source, YOU",
        "MUST reply that the question does not match any retrieved source. "
        "NEVER invent or add answer on your own, ONLY use the retrieved ",
        "sources. If none of the retrieved sources have an answer, ",
        "you MUST aknowledge that."
        "",
        "user",
        "Available sources:"
    ]

    for dt in data:
        path_content = f"path: {dt.file_path}, content: {dt.content}"
        sys_prompt.append(path_content)

    sys_prompt.append(f"\nUser's question: {query}\n")
    sys_prompt.append("Output: ")

    return "\n".join(sys_prompt)
