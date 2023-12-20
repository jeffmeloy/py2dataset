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
[req02] The main function shall:
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
    python_file_path: str,
    start_dir: str,
    model_config_path: str,
    questions_dict: Dict,
    use_llm: bool,
    output_dir: str,
    model_config: Dict = None,
    single_process_mode: bool = False,
    detailed_analysis: bool = False,
) -> None:
    """
    Processes a single Python file to generate question-answer pairs and instructions.

    Args:
        python_file_path (str): Path to the Python file.
        start_dir (str): Starting directory to search for Python files.
        model_config_path (str): Path to the model configuration file.
        questions_dict (Dict): Dictionary of questions to answer about the Python file.
        use_llm (bool): If True, use a Large Language Model for generating JSON answers.
        output_dir (str): Directory to save the output files.
        model_config (Dict): Configuration dictionary for the LLM.
        single_process_mode (bool): Use a single process if True. Defaults to False.
        detailed_analysis (bool): Perform detailed analysis if True. Defaults to False.
    """
    logging.info(f"Processing file: {python_file_path}")

    # need to define relative path as the having at least one parent directory
    # e.g relative patch should be equal to <last start_directory directory> / file name without path from python_file_path
    # Get the last directory in start_dir
    parent_dir = os.path.dirname(start_dir)
    relative_path = os.path.relpath(python_file_path, parent_dir)
    relative_path = Path(relative_path)
    base_name = ".".join(relative_path.parts)

    if not single_process_mode and use_llm:
        model_config = get_model(model_config_path)

    file_details = get_python_file_details(python_file_path)
    if not file_details:
        logging.error(f"Failed to get file details for {python_file_path}")
        return

    instruct_data = get_python_datasets(
        python_file_path,
        file_details,
        base_name,
        questions_dict,
        model_config,
        detailed_analysis,
    )

    if instruct_data:
        save_python_data(
            file_details, instruct_data, base_name, relative_path, output_dir
        )
    else:
        logging.error(f"Failed to get instruct data for {python_file_path}")


def py2dataset(
    start_dir: str = "",
    output_dir: str = "",
    questions_pathname: str = "",
    model_config_pathname: str = "",
    use_llm: bool = False,
    quiet: bool = False,
    single_process: bool = False,
    detailed: bool = False,
    html: bool = False,
) -> Dict[str, List[Dict]]:
    """
    Generates datasets by processing Python files within a specified directory.
    Args:
        start_dir (str): Starting directory for Python files. Defaults to current directory.
        output_dir (str): Directory to save the output files.
        questions_pathname (str): Path and filename of the questions file.
        model_config_pathname (str): Path and filename of the model configuration file.
        use_llm (bool): If True, use a Large Language Model for generating answers. Defaults to False.
        quiet_mode (bool): Reduce logging output if True. Defaults to False.
        single_process (bool): Use a single process for file processing if use_llm. Defaults to False.
        detailed (bool): Include detailed analysis if True. Defaults to False.
        html (bool): Generate HTML outputs if True. Defaults to False.
    Returns:
        Dict[str, List[Dict]]: Dictionary of generated datasets.
    """
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)

    sys.setrecursionlimit(3000)  # Set recursion limit higher for AST parsing

    start_dir = get_start_dir(start_dir)
    output_dir = get_output_dir(output_dir)
    questions_dict = get_questions(questions_pathname)

    # Load model configuration if LLM is used and single process mode is enabled
    model_config = (
        get_model(model_config_pathname) if use_llm and single_process else None
    )

    if not use_llm:
        single_process = True

    # Process each Python file in the directory
    for python_file_path in Path(start_dir).rglob("[!_]*.py"):
        # if use_llm is false or single_process is true then process the file within the current process

        if not use_llm and single_process:
            process_single_python_file(
                python_file_path,
                start_dir,
                model_config_pathname,
                questions_dict,
                use_llm,
                output_dir,
                model_config,
                single_process,
                detailed,
            )
        else:
            # Spawn new process for each file to manage memory and performance
            proc = Process(
                target=process_single_python_file,
                args=(
                    python_file_path,
                    start_dir,
                    model_config_pathname,
                    questions_dict,
                    use_llm,
                    output_dir,
                    None,
                    single_process,
                    detailed,
                ),
            )
            proc.start()
            proc.join()

    # Combine all the individual datasets into a single dictionary
    return combine_json_files(output_dir, html)


def clone_github_repo(url: str) -> str:
    """
    Clone repository or pull the latest changes and return local repository path.
    Args:
        url (str): The url of the github repository.
    Returns:
        str: The path to the cloned repository.
    """
    # Check valid Git repository
    try:
        command = f"git ls-remote {shlex.quote(url)}"
        subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        print(f"Invalid or inaccessible repository: {url}")
        return ""

    # Proceed with cloning or fetching
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
    return path


def main():
    """
    Command-line entry point for processing Python files and generating datasets.
    Optional command-line arguments:
    --start (str, optional): Starting directory for Python files or GitHub repository Python files. Defaults to current working directory.
    --output_dir (str, optional): Directory to write the output files. Defaults to ./dataset/.
    --questions_pathname (str, optional): Path and filename of the questions file. Defaults to ./py2dataset_questions.json.
    --model_config_pathname (str, optional): Path and filename of the model configuration file. Defaults to ./py2dataset_model_config.yaml.
    --use_llm (bool, optional): Use llm for generating JSON answers. Defaults to False.
    --quiet (bool, optional): Limit logging output. Defaults to False.
    --single_process (bool, optional): If True, only a single process will be used to process Python files. Defaults to False.
    --detailed (bool, optional): Include detailed analysis if True. Defaults to False.
    --html (bool, optional): Generate HTML output if True. Defaults to False.
    --I (str, optional): Interactive mode. Defaults to False.
    """
    if "--help" in sys.argv:
        print(__doc__)
        sys.exit()

    # helper function to convert user t or f input to bool
    def get_bool_from_input(input_str: str, current_value: bool) -> bool:
        if input_str.lower() in ["t", "true"]:
            return True
        elif input_str.lower() in ["f", "false"]:
            return False
        return current_value
        
    # Default values
    params = {
        'start': ".",
        'output_dir': "./dataset/",
        'questions_pathname': "./py2dataset_questions.json",
        'model_config_pathname': "./py2dataset_model_config.yaml",
        'use_llm': False,
        'quiet': False,
        'single_process': False,
        'detailed': False,
        'html': False,
        'I': False
    }

    # Process command-line arguments
    arg_string = " ".join(sys.argv[1:])  
    for arg in params:  
        if "--" + arg in arg_string:
            if isinstance(params[arg], bool):
                params[arg] = True  
                arg_string = arg_string.replace("--" + arg, "")  
            else:
                value_segment = arg_string.split("--" + arg + " ")[1]
                params[arg] = value_segment.split(" --")[0].strip('"')
                arg_string = arg_string.replace("--" + arg + " " + params[arg], "")

    # Interactive mode adjustments
    print("Interactive mode. Enter new values or press enter to keep.")
    if params['I']:
        for arg in params:
            user_input = input(f"{arg} [{params[arg]}]: ").strip()
            if isinstance(params[arg], bool):
                params[arg] = get_bool_from_input(user_input, params[arg])
            elif user_input:
                params[arg] = user_input
            print(f"{arg}: {params[arg]}")

    # Validate the start directory
    if not (os.path.isdir(params['start']) or params['start'].startswith("https://github.com/")):
        print(f"Invalid start directory '{params['start']}'. Using current working directory.")
        params['start'] = os.getcwd()

    # If the start directory is a github repository, clone it
    if params['start'].startswith("https://github.com/"):
        params['start'] = clone_github_repo(params['start'])

    # Call py2dataset with the parameters
    py2dataset(
        start_dir=params['start'],
        output_dir=params['output_dir'],
        questions_pathname=params['questions_pathname'],
        model_config_pathname=params['model_config_pathname'],
        use_llm=params['use_llm'],
        quiet=params['quiet'],
        single_process=params['single_process'],
        detailed=params['detailed'],
        html=params['html'],
    )

if __name__ == "__main__":
    main()

