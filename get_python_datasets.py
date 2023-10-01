"""
Generates JSON format question-answer pairs and instructions for a Python file
Requirements:
[req01] The `DatasetGenerator` class shall:
        a. Accept a Python file path (str), file details (Dict), base name (str), list of questions (List[Dict]), use_llm flag (bool), language model (object), and prompt (str) as input during instantiation.
        b. Initialize and store the Python file path, file details, base name, list of questions, use_llm flag, language model, and prompt as class attributes.
        c. Provide the `clean_and_get_unique_elements` method to clean an input string (str) and return a string of unique elements.
        d. Provide the `add_to_list` method to add a response (str) to a list (List[Dict]).
        e. Provide the `get_response_from_llm` method to retrieve a response from the language model.
        f. Provide the `process_question` method to process a question and add the generated response to the instruct_list.
        g. Provide the `process_question_type` method to process questions related to a file, function, class, or method.
        h. Provide the `generate` method to generate responses for all questions in the list and return the instruct_list.
[req02] The `get_python_datasets` function shall:
        a. Accept a Python file path (str), file details (Dict), base name (str), list of questions (List[Dict]), use_llm flag (bool), language model (object), and prompt (str) as input.
        b. Create an instance of the `DatasetGenerator` class using the provided input.
        c. Generate question-answer pairs and instructions using the `generate` method of the `DatasetGenerator` instance.
        d. Return the generated `instruct_list`.
"""
import logging
import re
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class DatasetGenerator:
    """
    Generate JSON formatted dictionary outputs for a Python file.
    Attributes:
        file_path (str): The path to the Python file.
        file_details (Dict[str, Any]): Details of the Python file.
        base_name (str): The base name of the Python file.
        questions (List[Dict[str, str]]): Questions for generating responses.
        instruct_list (List[Dict[str, str]]): Storage for generated instructions.
        question_mapping (Dict[str, str]): Mapping of question types to keys in file details.
        use_llm (bool): Flag indicating if a language model should be used.
        llm (object): The language model for generating responses.
        prompt (str): The prompt format for querying the language model.
    Methods:
        clean_and_get_unique_elements(input_str: str) -> str: 
            Clean and return unique elements from an input string.
        add_to_list(list_to_update: List[Dict], query: str, response: str, additional_field=None) -> List[Dict]: 
            Add response to the instruct list.
        get_response_from_llm(query: str, context: str) -> str:
            Get language model response to query for given context.
        process_question(question_type: str, question_id: str, query: str, context: str, info: Dict) -> None:
            Process question and add generated response to the instruct_list.
        process_question_type(question_type: str, question_id: str, question_text: str) -> None:
            Process question related to file, function, class, or method.
        generate() -> Tuple[List[Dict], List[Dict]]:
            Generate responses for all the questions and return the instruct_list.
    """
    def __init__(self, file_path: str, file_details: Dict, base_name: str, questions: List[Dict], llm: object, prompt: str) -> None:
        self.file_path = file_path
        self.file_details = file_details
        self.base_name = base_name
        self.questions = questions
        self.llm = llm
        self.prompt = prompt
        if self.llm is None:
            self.use_llm = False
        else:
            self.use_llm = True
        self.instruct_list = []
        self.question_mapping = {
            'file': 'file',
            'function': 'functions',
            'class': 'classes',
            'method': 'classes'
        }

    @staticmethod
    def clean_and_get_unique_elements(input_str: str) -> str:
        """
        Clean input string and return string of unique elements.
        Args:
            input_str (str): The input string to be cleaned.
        Returns:
            str: The cleaned string.
        """
        cleaned_elements = set(re.sub(r'[^\w\-_>\s:/.]', '', element.strip())
                               for element in re.sub(r'\s+', ' ', input_str).split(','))
        return ', '.join(cleaned_elements)

    @staticmethod
    def add_to_list(list_to_update: List[Dict], query: str, response: str, additional_field=None) -> List[Dict]:
        """
        Adds response to instruct list.
        Args:
            list_to_update (List[Dict]): The list to be updated.
            query (str): The query to be added.
            response (str): The response to be added.
            additional_field (str): The additional field to be added.
        Returns:
            List[Dict]: The updated list.
        """
        if response and response.strip() and response != 'None':
            list_to_update.append(
                {'instruction': query, 'input' : additional_field, 'output': response}
                if additional_field else
                {'question': query, 'answer': response}
            )
        return list_to_update

    def get_response_from_llm(self, query: str, context: str) -> str:
        """
        Get language model response to query for given context.
        Args:
            query (str): The query to be used for generating the response.
            context (str): The context to be used for generating the response.
        Returns:
            str: The generated response.
        """
        # Update the context with the selected instructions from the instruct_list
        excluded_instructions = ["What is the call code graph", "What is the docstring"]
        filtered_instruct_list = [item for item in self.instruct_list if not any(item['instruction'].startswith(prefix) for prefix in excluded_instructions)]
        past_instructs = "\n".join([f"Instruction: {item['instruction']} \nOutput: {item['output']}" for item in filtered_instruct_list])
        full_context = context + "\n" + "Here's some detail about this code:" + "\n" + past_instructs

        try:
            prompt = self.prompt.format(context=full_context, query=query)
            logging.info(f'Query: {query}')
            response = self.llm(prompt)
            logging.info(f'Response: {response}')
        except:
            logger.error('Failed to generate model response')
        return response

    def process_question(self, question_type: str, question_id: str, query: str, context: str, info: Dict) -> None:
        """
        Process question and add the generated response to the instruct_list.
        Args:
            question_type (str): The type of question to be processed.
            question_id (str): The ID of the question to be processed.
            query (str): The query to be processed.
            context (str): The context to be used for generating the response.
            info (Dict): The information of the Python file.
        Returns:
            None
        """
        if question_id.endswith('code_graph'):
            response = info.get(question_id, {})
        else:
            response = self.get_response_from_llm(query, context) if self.use_llm and question_id.endswith('purpose') else self.clean_and_get_unique_elements(str(info.get(question_id, '')))
        if response and response != 'None':
            response_str = str(response).strip()
            if response_str:
                self.instruct_list.append({'instruction': query, 'input': context, 'output': response_str})

    @staticmethod
    def get_string_from_info(info, item_type):
        if info[item_type]:
            items = [item.strip() for item in str(info[item_type]).split(',') if item]
            return ', '.join(items)
        return ''

    def process_question_type(self, question_type: str, question_id: str, question_text: str) -> None:
        """
        Process questions related to a file, function, class, or method.
        Args:
            question_type (str): The type of question to be processed.
            question_id (str): The ID of the question to be processed.
            question_text (str): The text of the question to be processed.
        Returns:
            None
        """
        if question_type == 'file':
            query = question_text.format(filename=self.base_name)
            context = self.file_details['file_info']['file_code']
            info = self.file_details['file_info']
            self.process_question(question_type, question_id, query, context, info)
        elif question_type == 'method':  
            for class_name, class_info in self.file_details['classes'].items():
                for key, method_info in class_info.items():
                    if key.startswith('class_method_'):
                        method_name = key[len('class_method_'):]
                        context = method_info['method_code']
                        mapping = {'class_name': class_name, 'method_name': method_name}
                        query = question_text.format(filename=self.base_name, **mapping)
                        self.process_question(question_type, question_id, query, context, method_info)
        else:  # if question_type == 'function' or question_type == 'class'
            for name, info in self.file_details[self.question_mapping[question_type]].items():
                context = info[f'{question_type}_code']
                mapping = {f'{question_type}_name': name}
                if question_id == f'{question_type}_purpose' and self.use_llm:
                    variables_string = self.get_string_from_info(info, f'{question_type}_variables')
                    inputs_string = self.get_string_from_info(info, f'{question_type}_inputs')
                    combined_string = ', '.join([s for s in [variables_string, inputs_string] if s])
                    mapping[f'{question_type}_variables'] = self.clean_and_get_unique_elements(combined_string)
                    # get methods to include in mapping for query
                    if question_type == 'class':
                        methods_string = self.get_string_from_info(info, f'{question_type}_methods')
                        mapping[f'{question_type}_methods'] = methods_string

                query = question_text.format(filename=self.base_name, **mapping)
                self.process_question(question_type, question_id, query, context, info)

    def generate(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate responses for all the questions and returns the instruct_list.
        Args:
            None
        Returns:
            Tuple[List[Dict], List[Dict]]: The generated question-answer pairs and instructions.
        """
        for question in self.questions:
            self.process_question_type(question['type'], question['id'], question['text'])
        return self.instruct_list


def get_python_datasets(file_path: str, file_details: Dict, base_name: str, questions: List[Dict], 
                        llm: object, prompt: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract information from a Python file and return it in JSON format.
    Args:
        file_path (str): The path to the Python file.
        file_details (Dict): The details of the file.
        base_name (str): The base Python code filename.
        questions (List[Dict]): The list of questions.
        llm (object): The language model to be used for generating responses.
        prompt (str): The prompt to be used for generating responses.
    Returns:
        Tuple[List[Dict], List[Dict]]: Extracted information in JSON format.
    """
    generator = DatasetGenerator(file_path, file_details, base_name, questions, llm, prompt)
    return generator.generate()
