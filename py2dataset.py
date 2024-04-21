import os
import sys
import logging
from pathlib import Path
from typing import Dict, List
from multiprocessing import Process
from git import Repo, GitCommandError

from get_python_file_details import get_python_file_details
from get_python_datasets import get_python_datasets
from get_params import get_questions, get_model, get_output_dir, get_start_dir
from save_output import combine_json_files, save_python_data


def process_single_python_file(params: Dict) -> None:
    """Processes a single Python file to generate question-answer pairs and instructions."""

    logging.info(f"Processing file: {params['python_pathname']}")
    file_details = get_python_file_details(params["python_pathname"])
    if not file_details:
        logging.error(f"Failed to get file details for {params['python_pathname']}")
        return

    if params["model_config"] is None and params["use_llm"]:
        params["model_config"] = get_model(params["model_config_pathname"])

    instruct_data = get_python_datasets(
        params["python_pathname"],
        file_details,
        params["relative_path"],
        params["questions"],
        params["model_config"],
        params["detailed"],
    )

    if instruct_data:
        save_python_data(
            file_details, instruct_data, params["relative_path"], params["output_dir"]
        )
    else:
        logging.error(f"Failed getting {params['python_pathname']} dataset")


def py2dataset(
    start: str = "",
    output_dir: str = "",
    questions_pathname: str = "",
    model_config_pathname: str = "",
    use_llm: bool = False,
    quiet: bool = False,
    single_process: bool = False,
    detailed: bool = False,
    html: bool = False,
    skip_regen: bool = False,
) -> Dict[str, List[Dict]]:
    """
    Generates datasets by processing Python files within a specified directory.
    Args:
        start (str, optional): Starting directory for Python files or GitHub repository Python files. Default: current working directory.
        output_dir (str, optional): Directory to write the output files. Default: ./dataset/.
        questions_pathname (str, optional): Path and filename of the questions file. Default: ./py2dataset_questions.json.
        model_config_pathname (str, optional): Path and filename of the model configuration file. Default: ./py2dataset_model_config.yaml.
        use_llm (bool, optional): Use llm to answer code purpose question. Default: False.
        quiet (bool, optional): Limit logging output. Default: False.
        single_process (bool, optional): Use a single process to process Python files if --use_llm. Default: False.
        detailed (bool, optional): Include detailed analysis. Default: False.
        html (bool, optional): Generate HTML output. Default: False.
        skip_regen (bool, optional): Skip regeneration of existing instruct.json files. Default: False.
    Returns:
        Dict[str, List[Dict]]: Generated datasets.
    """
    sys.setrecursionlimit(3000)  # Set recursion limit higher for AST parsing
    model_config = (
        get_model(model_config_pathname) if use_llm and single_process else None
    )

    logging.getLogger().setLevel(logging.WARNING if quiet else logging.INFO)

    params = {
        "output_dir": get_output_dir(output_dir),
        "model_config_pathname": model_config_pathname,
        "questions": get_questions(questions_pathname),
        "use_llm": use_llm,
        "model_config": model_config,
        "detailed": detailed,
    }

    exclude_dirs = ["env", "venv", "__pycache__", "build", "dist"]

    for python_pathname in Path(start).rglob("*.py"):
        if (
            any(dir in python_pathname.parts for dir in exclude_dirs)
            or python_pathname.name == "__init__.py"
        ):
            continue

        params["python_pathname"] = str(python_pathname)
        params["relative_path"] = Path(
            os.path.relpath(python_pathname, os.path.dirname(get_start_dir(start)))
        )
        instruct_pathname = Path(params["output_dir"]) / params[
            "relative_path"
        ].with_suffix(".py.instruct.json")

        if instruct_pathname.exists() and skip_regen:
            continue

        if model_config is None and use_llm:
            # Process each Python file in a separate process to manage memory
            process = Process(target=process_single_python_file, args=(params,))
            process.start()
            process.join()
            process.close()
        else:
            # Process all files using a single process
            process_single_python_file(params)

    return combine_json_files(output_dir, html, params["questions"])


def clone_github_repo(url: str) -> str:
    """Clone repository or fetch the latest changes and return local repository path."""
    try:
        repo_name = Path(url).stem
        repo_path = Path.cwd() / "githubrepos" / repo_name
        if repo_path.exists():
            repo = Repo(repo_path)
            repo.remote().fetch()
            repo.git.reset("--hard", repo.heads[0].commit)
        else:
            Repo.clone_from(url, repo_path)
        return str(repo_path)
    except GitCommandError as e:
        logging.info(f"Error processing repository {url}: {e}")
        return ""


def get_bool_from_input(input_str: str, current_value: bool) -> bool:
    """Return boolean value based on the user input."""
    if input_str.lower() in ["t", "true", "y", "yes"]:
        return True
    elif input_str.lower() in ["f", "false", "n", "no"]:
        return False
    return current_value


def main():
    """
    Command-line entry point for processing Python files and generating datasets.
    Usage:
        python py2dataset.py [options]
    Options:
        -h, --help: Show this help message and exit.
        --start: Starting directory for Python files or GitHub repository Python files. Default: current working directory.
        --output_dir: Directory to write the output files. Default: ./dataset/.
        --questions_pathname: Path and filename of the questions file. Default: ./py2dataset_questions.json.
        --model_config_pathname: Path and filename of the model configuration file. Default: ./py2dataset_model_config.yaml.
        --use_llm: Use llm to answer code purpose question. Default: False.
        --quiet: Limit logging output. Default: False.
        --single_process: Use a single process to process Python files if --use_llm. Default: False.
        --detailed: Include detailed analysis. Default: False.
        --html: Generate HTML output. Default: False.
        --skip_regen: Skip regeneration of existing instruct.json files. Default: False.
        --I: Interactive mode to enter new values.
    """
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__)
        sys.exit()

    params = {
        "start": ".",
        "output_dir": "./dataset/",
        "questions_pathname": "./py2dataset_questions.json",
        "model_config_pathname": "./py2dataset_model_config.yaml",
        "use_llm": False,
        "quiet": False,
        "single_process": False,
        "detailed": False,
        "html": False,
        "skip_regen": False,
        "I": False,
    }

    arg_string = " ".join(sys.argv[1:])
    for arg, value in params.items():
        if f"--{arg}" in arg_string:
            if isinstance(value, bool):
                params[arg] = True
                arg_string = arg_string.replace(f"--{arg}", "")
            else:
                value_segment = arg_string.split(f"--{arg} ")[1]
                params[arg] = value_segment.split(" --")[0].strip('"')
                arg_string = arg_string.replace(f"--{arg} {params[arg]}", "")

    if params.pop("I"):
        print("Interactive mode, enter new values or press enter to keep.")
        for arg, value in params.items():
            user_input = input(f"{arg} [{value}]: ").strip()
            params[arg] = (
                get_bool_from_input(user_input, value)
                if isinstance(value, bool)
                else user_input or value
            )
            print(f"{arg}: {params[arg]}")

    params["start"] = (
        clone_github_repo(params["start"])
        if params["start"].startswith("https://github.com/")
        else params["start"]
        if os.path.isdir(params["start"])
        else os.getcwd()
    )

    py2dataset(**params)


if __name__ == "__main__":
    main()
