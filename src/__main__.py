import os
import fire
from rich.panel import Panel
from rich.console import Console
from src.indexer.serialization import index
from src.retriever.bm25 import search
from src.retriever.search_datasets import search_dataset
from src.generator.generation import answer, answer_dataset
from src.evaluator.evaluation import intersection_over_union as evaluate


console = Console()


if __name__ == "__main__":
    try:
        commands = {
            "index": index,
            "search": search,
            "search_dataset": search_dataset,
            "answer": answer,
            "answer_dataset": answer_dataset,
            "evaluate": evaluate
        }
        os.system('clear')

        console.print(Panel.fit("[bold green]Welcome to "
                                "RAG-against-the-machine"
                                "[/bold green] [bold blue] by sbelabbe"
                                "[/bold blue] [bold green]"))
        fire.Fire(commands)
    except ValueError as e:
        console.print(f"[bold underline red]{e}")
    except FileNotFoundError as e:
        console.print("[bold underline red][ERROR]: No file/directory "
                      f"detected: {e.filename}")
    except TypeError as e:
        console.print(f"[bold underline red]{e}")
    except KeyboardInterrupt:
        console.print("[bold underline red][ERROR]: You killed the "
                      "program!")
    except OSError as e:
        console.print(f"[bold underline red]{e}")
