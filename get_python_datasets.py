"""
Generates JSON format question-answer pairs and instructions for a Python file
Requirements:
[req01] The `DatasetGenerator` class shall:
        a. Accept Python file path, file details, base name, list of questions,
        and model configuration as input during instantiation.
        b. Initialize and store Python file path, file details, base name,
        question list, llm, and use_llm flag as class attributes.
        d. Provide `get_response_from_llm` method to retrieve llm response.
        e. Provide `get_detailed_response` method to generate detailed responses.
        f. Provide `get_code_qa` method to get code responses from instruct_list.
        g. Provide `process_question` method to process each question, generate
        corresponding responses, and add to the instruct_list.
        h. Provide `process_question_type` method to process questions
        related to the file, functions, classes, and methods.
        i. Provide `generate` method to generate responses for all questions
        and return the instruct_list.
        j. Internally manage question mapping to file details.
[req02] The `get_python_datasets` function shall:
        a. Accept a Python file path, file details, base name, questions list,
        and model config as input.
        b. Instantiate `DatasetGenerator` class using the provided input.
        c. Generate instruct_list using `DatasetGenerator` `generate` method.
        d. Return the generated `instruct_list`.
[req03] The `get_unique_elements` function shall:
        a. Clean an input string (str) and return a string of unique elements.
"""

import logging
import re
import math
import yaml


def get_unique_elements(input_str: str) -> str:
    """
    Clean an input string (str) and return a string of unique elements.
    Args:
        input_str (str): Input string.
    Returns:
        str: String of unique elements.
    """
    def element_generator(input_str):
        start, brace_level = 0, 0
        for i, char in enumerate(input_str):
            if char in "{}":
                brace_level += 1 if char == "{" else -1
            elif char == "," and brace_level == 0:
                yield input_str[start:i].strip("'\" ")
                start = i + 1
        yield input_str[start:].strip("'\" ")
    input_str = input_str.strip("[]'\"")
    cleaned_elements = [element for element in element_generator(input_str) if element]
    return ", ".join(cleaned_elements)


class DatasetGenerator:
    """
    Generate JSON formatted dictionary outputs for a Python file.
    Args:
        file_path (str): Path to the Python file.
        file_details (dict): Dictionary containing details of the Python file.
        base_name (str): Base name of the Python file.
        questions (list): List of questions to be answered for the Python file.
        model_config (dict): Dictionary containing the model configuration.
        detailed (bool): Flag to generate detailed responses.
    Attributes:
        file_path (str): Path to the Python file.
        file_details (dict): Dictionary containing details of the Python file.
        base_name (str): Base name of the Python file.
        questions (list): List of questions to be answered for the Python file.
        model_config (dict): Dictionary containing the model configuration.
        llm (obj): Language model object.
        use_llm (bool): Flag to use language model.
        detailed (bool): Flag to generate detailed responses.
        instruct_list (list): List of instructions and responses.
        question_mapping (dict): Mapping of question types to file details.
        code_qa_list (list): List of code questions and responses.
        code_qa_response (str): Response for the code_qa_dict.
        code_qa_dict (dict): Dictionary containing code questions and responses.
    """

    def __init__(
        self,
        file_path: str,
        file_details: dict,
        base_name: str,
        questions: list[dict],
        model_config: dict,
        detailed: bool,
    ) -> None:
        """Initialize the DatasetGenerator class."""
        self.file_path = file_path
        self.file_details = file_details
        self.base_name = base_name
        self.questions = questions
        self.model_config = model_config
        self.llm = model_config["model"] if model_config else None
        self.use_llm = bool(model_config)
        self.detailed = detailed if self.use_llm else False
        self.instruct_list = []
        self.question_mapping = {
            "file": "file",
            "function": "functions",
            "class": "classes",
            "method": "classes",
        }
        self.code_qa_list = []
        self.code_qa_response = ""

    def format_response(self):
        """Format the response for the code_qa_dict."""
        self.code_qa_response = (
            re.sub(
                r"\n\s*\n",
                "\n",
                yaml.dump(
                    self.code_qa_dict,
                    Dumper=yaml.SafeDumper,
                    width=float("inf"),
                    sort_keys=False,
                    default_flow_style=False,
                    indent=2,
                ),
            )
            .replace("''", "'")
            .strip('"')
            .strip("'")
            .strip()
        )
        self.file_details["file_info"]["code_qa_response"] = self.code_qa_response

    def get_response_from_llm(self, query: str, context: str) -> str:
        """
        Get language model response to query for given context.
        Args:
            query (str): Query to be answered.
            context (str): Context for the query.
        Returns:
            str: Language model response.
        """
        context_strategies = [
            lambda: str(context),
            lambda: f"```python\n{self.file_details['file_info']['file_code_simplified']}\n```",
            lambda: f"```python\n{self.get_info_string(self.file_details['file_info'], 'file_summary')}\n```",
            lambda: "",
        ]

        max_context_length = self.model_config["inference_model"]["model_params"][
            "context_length"
        ]
        prompt_template = self.model_config["prompt_template"].format(
            system_prompt=self.model_config["system_prompt"],
            instruction_prompt=self.model_config["instruction_prompt"],
        )

        for strategy in context_strategies:
            context = strategy()
            prompt = prompt_template.format(
                context=context, query=query, code_objects=self.code_qa_response
            )
            context_size = len(self.llm.tokenize(prompt))
            logging.info(f"***Context Size: {context_size}")
            if context_size <= 0.70 * max_context_length:
                break
        else:
            err_msg = "Model response failed, increase py2dataset_model_config.yaml context_length"
            logging.error(f"{err_msg} > {math.ceil(context_size/0.70)}")
            return ""

        context = self.code_qa_list if context == "" else context
        response = ""

        try:
            response = (
                re.sub(r"\n\s*\n", "\n\n", self.llm(prompt))
                .replace("<|im_end|>", "")
                .replace("</|im_end|>", "")
            )
            response = "\n".join(line.lstrip() for line in response.split("\n"))
            logging.info(f"***Overall Response: {response}")
        except Exception as error:
            logging.error(f"Failed to generate model response: {error}")

        if self.detailed:
            self.get_detailed_response(context, response)

        basename = str(self.base_name).replace("\\", "/")
        self.code_qa_dict = {basename: self.code_qa_dict}
        self.code_qa_dict[basename] = {
            "Code Documentation": [response],
            **self.code_qa_dict[basename],
        }
        self.format_response()
        self.file_details["file_info"]["purpose"] = response
        return response

    def get_detailed_response(self, context: str, response: str) -> None:
        """
        Generate detailed responses for code objects.
        Args:
            context (str): Context for the response.
            response (str): Response from the model.
        """
        for item in self.code_qa_list:
            try:
                instruct_key = list(item.keys())[0]
                instruct_value = list(item.values())[0]
                query = f"Describe the Purpose and Significance of these {instruct_key}: [{instruct_value}] and Explain what each of these {instruct_key} does in the code."
                prompt = (
                    self.model_config["prompt_template"]
                    .format(
                        system_prompt=self.model_config["system_prompt"],
                        instruction_prompt=self.model_config["instruction_prompt"],
                    )
                    .format(
                        context=f"{context}/nCode Summary:/n{response}",
                        query=f"{query}",
                        code_objects=f"{instruct_value}",
                    )
                )
                item_response = (
                    re.sub(r"\n\s*\n", "\n\n", self.llm(prompt))
                    .replace("<|im_end|>", "")
                    .replace("</|im_end|>", "")
                )
                logging.info(f"\n***Itemized Response: {query}\n{item_response}")
                for item in self.instruct_list:
                    if item["instruction"].startswith(instruct_key):
                        output = f"\n\nPurpose and Significance:\n{item_response}"
                        item["output"] += output

                if "`" in instruct_key:
                    dict_key1 = instruct_key.split("`")[1]
                    dict_key2 = instruct_key.split()[0]
                else:
                    dict_key1 = instruct_key
                    dict_key2 = ""

                purpose_dict = {"Purpose": item_response.strip()}
                if dict_key1 not in self.code_qa_dict:
                    self.code_qa_dict[dict_key1] = {}
                elif not isinstance(self.code_qa_dict[dict_key1], dict):
                    value = self.code_qa_dict[dict_key1]
                    self.code_qa_dict[dict_key1] = {"Value": value}
                if dict_key2:
                    if not isinstance(self.code_qa_dict[dict_key1], dict):
                        self.code_qa_dict[dict_key1] = {}
                    if dict_key2 not in self.code_qa_dict[dict_key1] or not isinstance(
                        self.code_qa_dict[dict_key1][dict_key2], dict
                    ):
                        value = self.code_qa_dict[dict_key1].get(dict_key2, "")
                        self.code_qa_dict[dict_key1][dict_key2] = {"Value": value}
                    self.code_qa_dict[dict_key1][dict_key2].update(purpose_dict)
                else:
                    self.code_qa_dict[dict_key1].update(purpose_dict)
            except Exception as error:
                logging.error(f"Failed to generate detailed response: {error}")

    def get_code_qa(self):
        """Get code responses from the instruct_list and update."""
        excluded = {"Call code graph", "Docstring"}
        self.code_qa_list = []
        responses = {}
        for item in self.instruct_list:
            instruction, output = (
                item["instruction"].split(" in Python file:")[0],
                item["output"],
            )
            if not any(instruction.startswith(prefix) for prefix in excluded):
                self.code_qa_list.append({instruction: output})
                if "`" in instruction:
                    code_object, code_type = (
                        instruction.split("`")[1],
                        instruction.split()[0],
                    )
                    responses.setdefault(code_object, []).append((code_type, output))
                else:
                    responses.setdefault(instruction, []).append((instruction, output))

        self.code_qa_dict = {}
        for code_object, type_responses in responses.items():
            self.code_qa_dict[code_object] = {}
            for code_type, response in type_responses:
                if code_object == code_type:
                    self.code_qa_dict[code_object] = response
                else:
                    self.code_qa_dict.setdefault(code_object, {})[code_type] = response

        if not self.use_llm:
            basename = str(self.base_name).replace("\\", "/")
            self.code_qa_dict = {basename: self.code_qa_dict}

        self.format_response()

    def process_question(
        self, question_id: str, query: str, context: str, info: dict
    ) -> None:
        """Process question and add the generated response to the instruct_list."""
        if question_id.endswith(("code_graph", "docstring")):
            response = info.get(question_id, {})
        elif question_id.endswith("file_purpose"):  # file_purpose is last question
            self.get_code_qa()
            response = (
                self.get_response_from_llm(query, context) if self.use_llm else ""
            )
        else:
            response = get_unique_elements(str(info.get(question_id, "")))

        if response and response != "None":
            response = str(response).strip()
            self.instruct_list.append(
                {"instruction": query, "input": context, "output": response}
            )

    @staticmethod
    def get_info_string(info: dict, item_type: str) -> str:
        """Get string from info dictionary."""
        return ", ".join(
            [item.strip() for item in str(info.get(item_type, "")).split(",") if item]
        )

    def process_question_type(
        self, question_type: str, question_id: str, question_text: str
    ) -> None:
        """
        Process questions related to a file, function, class, or method.
        Args:
            question_type (str): Type of question.
            question_id (str): Question ID.
            question_text (str): Question text.
        """
        if question_type == "file":
            query = question_text.format(filename=self.base_name)
            info = self.file_details["file_info"]
            context = f"```python\n{self.file_details['file_info']['file_code']}\n```"
            self.process_question(question_id, query, context, info)
        elif question_type == "method":
            for class_name, class_info in self.file_details["classes"].items():
                for key, method_info in class_info.items():
                    if key.startswith("class_method_"):
                        method_name = f"{class_name}.{key[len('class_method_'):]}"
                        context = f"```python\n{method_info['method_code']}\n```"
                        mapping = {"class_name": class_name, "method_name": method_name}
                        query = question_text.format(filename=self.base_name, **mapping)
                        self.process_question(question_id, query, context, method_info)
        else:  # question_type is 'function' or 'class'
            for name, info in self.file_details[
                self.question_mapping[question_type]
            ].items():
                context = f"```python\n{info[f'{question_type}_code']}\n```"
                mapping = {f"{question_type}_name": name}
                if question_id == f"{question_type}_purpose" and self.use_llm:
                    variables = self.get_info_string(info, f"{question_type}_variables")
                    inputs = self.get_info_string(info, f"{question_type}_inputs")
                    combined = ", ".join(filter(None, [variables, inputs]))
                    mapping[f"{question_type}_variables"] = get_unique_elements(
                        combined
                    )
                    if question_type == "class":
                        mapping[f"{question_type}_methods"] = self.get_info_string(
                            info, f"{question_type}_methods"
                        )
                query = question_text.format(filename=self.base_name, **mapping)
                self.process_question(question_id, query, context, info)

    def generate(self) -> tuple[list[dict], list[dict]]:
        """Generate responses for all the questions and returns the instruct_list."""
        for question in self.questions:
            self.process_question_type(
                question["type"], question["id"], question["text"]
            )
        self.instruct_list.sort(key=lambda x: len(x["input"]), reverse=True)
        return self.instruct_list


def get_python_datasets(
    file_path: str,
    file_details: dict,
    base_name: str,
    questions: list[dict],
    model_config: dict,
    detailed: bool,
) -> tuple[list[dict], list[dict]]:
    """
    Extract information from a Python file and return it in JSON format.
    Args:
        file_path (str): Path to the Python file.
        file_details (dict): Dictionary containing details of the Python file.
        base_name (str): Base name of the Python file.
        questions (list): List of questions to be answered for the Python file.
        model_config (dict): Dictionary containing the model configuration.
        detailed (bool): Flag to generate detailed responses.
    Returns:
        tuple: Tuple containing the instruct_list and code_qa_dict.
    """
    return DatasetGenerator(
        file_path, file_details, base_name, questions, model_config, detailed
    ).generate()
