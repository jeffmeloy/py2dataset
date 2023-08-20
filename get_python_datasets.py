"""
Generates JSON format question-answer pairs and instructions for a Python file
Requirements:
[req01] The `DatasetGenerator` class shall:
        a. Accept a Python file path (str), file details (Dict), base name (str), list of questions (List[Dict]), use_summary flag (bool), use_llm flag (bool), language model (object), and prompt (str) as input during instantiation.
        b. Initialize and store the Python file path, file details, base name, list of questions, use_summary flag, use_llm flag, language model, and prompt as class attributes.
        c. Provide the `clean_and_get_unique_elements` method to clean an input string (str) and return a string of unique elements.
        d. Provide the `add_to_list` method to add a response (str) to a list (List[Dict]).
        e. Provide the `get_response_from_llm` method to retrieve a response from the language model.
        f. Provide the `process_items` method to process questions related to the purpose of a variable.
        g. Provide the `process_question` method to process a question and add the generated response to the qa_list and instruct_list.
        h. Provide the `process_question_type` method to process questions related to a file, function, class, or method.
        i. Provide the `generate` method to generate responses for all questions in the list and return the qa_list and instruct_list.
[req02] The `get_python_datasets` function shall:
        a. Accept a Python file path (str), file details (Dict), base name (str), list of questions (List[Dict]), use_summary flag (bool), use_llm flag (bool), language model (object), and prompt (str) as input.
        b. Create an instance of the `DatasetGenerator` class using the provided input.
        c. Generate question-answer pairs and instructions using the `generate` method of the `DatasetGenerator` instance.
        d. Return the generated `qa_list` and `instruct_list`.
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
    A class used to generate JSON formatted dictionary outputs for a Python 
    file.
    Attributes:
        file_path (str): The path to the Python file.
        file_details (Dict[str, Any]): Details of the Python file.
        base_name (str): The base name of the Python file.
        questions (List[Dict[str, str]]): Questions for generating responses.
        qa_list (List[Dict[str, str]]): Storage for generated question-answer pairs.
        instruct_list (List[Dict[str, str]]): Storage for generated instructions.
        question_mapping (Dict[str, str]): Mapping of question types to keys in file details.
        use_llm (bool): Flag indicating if a language model should be used.
        llm (object): The language model for generating responses.
        use_summary (bool): Flag indicating if file summary should be used.
        prompt (str): The prompt format for querying the language model.
    Methods:
        clean_and_get_unique_elements(input_str: str) -> str: 
            Clean and return unique elements from an input string.
        add_to_list(list_to_update: List[Dict], query: str, response: str, additional_field=None) -> List[Dict]: 
            Add a response to a list.
        get_response_from_llm(query: str, context: str) -> str:
            Retrieve a response from the language model.
        get_variable_purpose(question_id: str, question_text: str, base_name: str, name: str, info: Dict, context: str, variable_type: str) -> None: 
            Process questions about a variable's purpose.
        process_question(question_id: str, query: str, context: str, info) -> None: 
            Process a question and add the response to lists.
        process_file_question(question_id: str, question_text: str) -> None:
            Process questions related to a file.
        process_func_class_question(question_type: str, question_id: str, question_text: str) -> None: 
            Process questions about a function or class.
        generate() -> Tuple[List[Dict], List[Dict]]: 
            Generate responses for all questions and return them.
    """
    def __init__(self, file_path: str, file_details: Dict, base_name: str, questions: List[Dict], use_summary: bool, use_llm: bool, llm: object, prompt: str) -> None:
        self.file_path = file_path
        self.file_details = file_details
        self.base_name = base_name
        self.questions = questions
        self.qa_list = []
        self.instruct_list = []
        self.question_mapping = {
            'file': 'file',
            'function': 'functions',
            'class': 'classes',
            'method': 'classes'
        }
        self.use_summary = use_summary
        self.use_llm = use_llm
        self.llm = llm
        self.prompt = prompt
        if not self.use_llm or self.llm is None:
            self.use_llm = False
            self.llm = None


    @staticmethod
    def clean_and_get_unique_elements(input_str: str) -> str:
        """
        Cleans an input string and returns a string of unique elements.
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
        Adds a response to a list.
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
        Gets a response from the language model.
        Args:
            query (str): The query to be used for generating the response.
            context (str): The context to be used for generating the response.
        Returns:
            str: The generated response.
        """
        response = ''
        try:
            prompt = self.prompt.format(context=context, query=query)
            logging.info(f'Query: {query}')
            response = self.llm(prompt)
            logging.info(f'Response: {response}')
        except:
            logger.error('Failed to generate model response')
        return response


    def process_items(self, question_type: str, question_id: str, question_text: str, base_name: str, name: str, info: Dict, context: str, item_type: str) -> None:
        """
        Processes questions related to the purpose of a variable.
        Args:
            question_type (str): The type of question to be processed.
            question_id (str): The ID of the question to be processed.
            question_text (str): The text of the question to be processed.
            base_name (str): The base name of the Python file.
            name (str): The name of the variable.
            info (Dict): The information of the variable.
            context (str): The context to be used for generating the response.
            item_type (str): The type of the variable.
        Returns:
            None
        """
        if info[item_type]:
            items = [item.strip() for item in self.clean_and_get_unique_elements(str(info[item_type])).split(',') if item]
            itemstring = ', '.join(items)
            query = question_text.format(filename=base_name, **{f'{question_type.split("_")[0]}_name': name, f'{question_type.split("_")[0]}_variables': itemstring})
            self.process_question(question_type, question_id, query, context, info)


    def process_question(self, question_type: str, question_id: str, query: str, context: str, info: Dict) -> None:
        """
        Processes a question and adds the generated response to the qa_list and instruct_list.
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
            response_str = str(response)
            response_str = response_str.strip()
            if response_str:
                self.qa_list.append({'question': query, 'answer': response_str})
                if question_type == 'file' and self.use_summary:
                    context = info['file_summary']
                self.instruct_list.append({'instruction': query, 'input': context, 'output': response_str})

    def process_question_type(self, question_type: str, question_id: str, question_text: str) -> None:
        """
        Processes questions related to a file, function, class, or method.
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
        else:
            for name, info in self.file_details[self.question_mapping[question_type]].items():
                context = info[f'{question_type}_code']
                mapping = {f'{question_type}_name': name}
                if question_id == f'{question_type}_variable_purpose' and self.use_llm:
                    self.process_items(question_type, question_id, question_text, self.base_name, name, info, context, f'{question_type}_variables')
                elif question_id != f'{question_type}_variable_purpose':
                    query = question_text.format(filename=self.base_name, **mapping)
                    self.process_question(question_type, question_id, query, context, info)

    def generate(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Generates responses for all the questions and returns the qa_list and instruct_list.
        Args:
            None
        Returns:
            Tuple[List[Dict], List[Dict]]: The generated question-answer pairs and instructions.
        """
        for question in self.questions:
            question_id = question['id']
            question_text = question['text']
            question_type = question['type']
            self.process_question_type(question_type, question_id, question_text)
        return self.qa_list, self.instruct_list


def get_python_datasets(file_path: str, file_details: Dict, base_name: str, questions: List[Dict], use_summary: bool,
                        use_llm: bool, llm: object, prompt: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract information from a Python file and return it in JSON format.
    Args:
        file_path (str): The path to the Python file.
        file_details (Dict): The details of the file.
        base_name (str): The base Python code filename.
        questions (List[Dict]): The list of questions.
        llm (object): The language model to be used for generating responses.
        prompt (str): The prompt to be used for generating responses.
        use_llm (bool): Whether to use the language model.
        use_summary (bool): Whether to use the file summary instead of the file code for the instruct_list input context.
    Returns:
        Tuple[List[Dict], List[Dict]]: Extracted information in JSON format.
    """
    generator = DatasetGenerator(file_path, file_details, base_name, questions, use_summary, use_llm, llm, prompt)
    return generator.generate()
