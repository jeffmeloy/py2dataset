"""
For each Python file within a given directory, this module is designed to generate, save, 
and return datasets that include responses to questions about the code. 
Requirements:
[req00] The process_single_python_file function shall:
    a. Accept parameters for Python file path, start directory, model config 
       pathname, questions dict, use of LLM, and output dir.
    b. If 'use_llm' is True, use 'get_model' to instantiate LLM model config.
    c. Use 'get_python_file_details' to retrieve Python file info.
    d. Use 'get_python_datasets' to acquire instruct.json datasets.
    e. Use 'save_python_data' to store file details and instruct.json data.
[req01] The py2dataset function shall:
    a. Accept parameters for start directory, output dir, questions path, model
       config path, use of LLM, and quiet mode.
    b. Adjust logging level based on 'quiet'.
    c. Use current working dir if no valid start dir is provided.
    d. Get output dir with 'get_output_dir'.
    e. Retrieve questions dict with 'get_questions'.
    f. Search for Python files using 'rglob', excluding those starting with "_".
    g. For each Python file, spawn a child process with 'process_single_python_file'
       to get file details and instruct.json data, if single_process is False.
    h. Combine instruct.json files with 'combine_json_files'.
    i. Return datasets.
[req02] The clone_github_repo function shall:
    a. Accept a url.
    b. Clone repository or fetch the latest changes.
    c. Return local repository path.
[req03] The main function shall:
    a. Accept and process command-line args.
    b. Determine py2dataset parameters based on processed arguments.
    c. Call py2dataset with derived parameters.
"""
import sys
import logging
from pathlib import Path
from typing import Dict, List
from multiprocessing import Process
import subprocess
import os
import git
import shlex

from get_python_file_details import get_python_file_details
from get_python_datasets import get_python_datasets
from get_params import (
    get_questions,
    get_model,
    get_output_dir,
    get_start_dir,
)
from save_output import combine_json_files, save_python_data


def process_single_python_file(
    python_pathname: str,
    start: str,
    output_dir: str,
    model_config_pathname: str,
    questions: Dict,
    use_llm: bool,
    model_config: Dict,
    detailed: bool,
    skip_regen: bool,
) -> None:
    """
    Processes a single Python file to generate question-answer pairs and instructions.
    Args:
        python_pathname (str): Path to the Python file.
        start (str): Starting directory for Python files.
        output_dir (str): Directory to write the output files.
        model_config_pathname (str): Path and filename of the model configuration file.
        questions (Dict): Dictionary of questions to answer about the Python file.
        use_llm (bool): Use llm to answer code purpose question.
        model_config (Dict): Configuration dictionary for the LLM.
        detailed (bool): Perform detailed analysis if True.
        skip_regen (bool): Skip regeneration of existing instruct.json files.
    """
    relative_path = Path(os.path.relpath(python_pathname, os.path.dirname(start)))
    base_pathname = Path(output_dir) / relative_path
    instruct_pathname = base_pathname.with_suffix(".py.instruct.json")
    if instruct_pathname.exists() and skip_regen:
        return

    logging.info(f"Processing file: {python_pathname}")
    if model_config is None and use_llm:
        model_config = get_model(model_config_pathname)

    file_details = get_python_file_details(python_pathname)
    if not file_details:
        logging.error(f"Failed to get file details for {python_pathname}")
        return

    instruct_data = get_python_datasets(
        python_pathname,
        file_details,
        relative_path,
        questions,
        model_config,
        detailed,
    )

    if instruct_data:
        save_python_data(file_details, instruct_data, relative_path, output_dir)
    else:
        logging.error(f"Failed getting {python_pathname} dataset")

    del instruct_data, file_details, model_config


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
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)

    sys.setrecursionlimit(3000)  # Set recursion limit higher for AST parsing

    model_config = None
    if use_llm and single_process:
        model_config = get_model(model_config_pathname)

    params = {
        "python_pathname": "",
        "start": get_start_dir(start),
        "output_dir": get_output_dir(output_dir),
        "model_config_pathname": model_config_pathname,
        "questions": get_questions(questions_pathname),
        "use_llm": use_llm,
        "model_config": model_config,
        "detailed": detailed,
        "skip_regen": skip_regen,
    }

    for python_pathname in Path(start).rglob("[!_]*.py"):
        params["python_pathname"] = str(python_pathname)
        # process each python file in a separate process to manage memory
        if params["model_config"] is None and params["use_llm"]:
            proc = Process(target=process_single_python_file, kwargs=params)
            proc.start()
            proc.join()
        else:  # or process all files using use a single process
            process_single_python_file(**params)

    return combine_json_files(output_dir, html)


def clone_github_repo(url: str) -> str:
    """
    Clone repository or pull the latest changes and return local repository path.
    Args:
        url (str): The url of the github repository.
    Returns:
        str: The path to the cloned repository.
    """
    try:
        command = f"git ls-remote {shlex.quote(url)}"
        subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        repo_name = url.split("/")[-1]
        githubrepos_dir = os.path.join(os.getcwd(), "githubrepos")
        os.makedirs(githubrepos_dir, exist_ok=True)
        path = os.path.join(githubrepos_dir, repo_name)
        if not os.path.exists(path):
            git.Repo.clone_from(url, path)
        else:
            repo = git.Repo(path)
            with repo.git.custom_environment(
                GIT_SSH_COMMAND="ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no"
            ):
                repo.git.fetch()
                default_branch = repo.head.reference.tracking_branch().remote_head
                repo.git.reset("--hard", default_branch)
    except subprocess.CalledProcessError:
        logging.info(f"Invalid or inaccessible repository: {url}")
        path = ""

    return path


def main():
    """
    Command-line entry point for processing Python files and generating datasets.
    Optional command-line arguments:
    --start (str, optional): Starting directory for Python files or GitHub repository Python files. Default: cwd.
    --output_dir (str, optional): Directory to write the output files. Default: ./dataset/.
    --questions_pathname (str, optional): Path and filename of the questions file. Default: ./py2dataset_questions.json.
    --model_config_pathname (str, optional): Path and filename of the model configuration file. Default: ./py2dataset_model_config.yaml.
    --use_llm (bool, optional): Use llm to answer code purpose question. Default: False.
    --quiet (bool, optional): Limit logging output. Default: False.
    --single_process (bool, optional): Use a single process to process Python files if --use_llm. Default: False.
    --detailed (bool, optional): Include detailed analysis. Default: False.
    --html (bool, optional): Generate HTML output. Default: False.
    --I (str, optional): Enable interactive mode. Default: False.
    --skip_regen (str, optional): Skip regeneration of existing instruct.json files. Default: False.
    --help: Display help message.
    """
    if "--help" in sys.argv:
        print(__doc__)
        sys.exit()

    # return boolean value based on the user input
    def get_bool_from_input(input_str: str, current_value: bool) -> bool:
        if input_str.lower() in ["t", "true", "y", "yes"]:
            return True
        elif input_str.lower() in ["f", "false", "n", "no"]:
            return False
        return current_value

    # Defaults
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
    for arg in params: # parse command-line arguments
        if "--" + arg in arg_string:
            if isinstance(params[arg], bool):
                params[arg] = True
                arg_string = arg_string.replace("--" + arg, "")
            else:
                value_segment = arg_string.split("--" + arg + " ")[1]
                params[arg] = value_segment.split(" --")[0].strip('"')
                arg_string = arg_string.replace("--" + arg + " " + params[arg], "")

    if params["I"]: # query user for parameters to change
        print("Interactive mode, enter new values or press enter to keep.")
        for arg in params:
            if arg != "I":
                user_input = input(f"{arg} [{params[arg]}]: ").strip()
                if isinstance(params[arg], bool):
                    params[arg] = get_bool_from_input(user_input, params[arg])
                elif user_input:
                    params[arg] = user_input
                print(f"{arg}: {params[arg]}")
    params.pop("I")

    if params["start"].startswith("https://github.com/"):
        params["start"] = clone_github_repo(params["start"])
    elif not os.path.isdir(params["start"]):
        print(f"'{params['start']}' Invalid. Using current working directory.")
        params["start"] = os.getcwd()

    py2dataset(**params)


if __name__ == "__main__":
    main()
