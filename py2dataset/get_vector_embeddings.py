"""
This module processes Python code and generated questions and instructions to create vector embeddings. 
It uses a pretrained model to generate the embeddings.
Requirements:
    [req001] The get_vector_embeddings function shall accept a dictionary representing file details, a list of QA pairs, and a list of instructions.
    [req002] The get_vector_embeddings function shall use a pretrained embedding model to generate vector embeddings for the provided inputs.
    [req003] The get_vector_embeddings function shall return three lists of vector embeddings, one for each input.
    [req004] The get_vector_embeddings function shall process each Python file detail in chunks to generate embeddings, if an AST is provided.
    [req005] The EmbeddingGenerator class shall be initialized with a tokenizer and model from a pretrained embedding model.
    [req006] The EmbeddingGenerator class shall generate vector embeddings for a given piece of text using its get_embeddings method.
    [req007] The get_embeddings method of the EmbeddingGenerator class shall accept a piece of text.
    [req008] The get_embeddings method of the EmbeddingGenerator class shall use the tokenizer and model of the EmbeddingGenerator to generate a vector embedding for the provided text.
    [req009] The get_embeddings method of the EmbeddingGenerator class shall return the generated vector embedding.
    [req010] The get_ast_embeddings function shall accept an AST and an EmbeddingGenerator.
    [req011] The get_ast_embeddings function shall generate embeddings for the provided AST using the EmbeddingGenerator.
    [req012] The get_ast_embeddings function shall recursively generate embeddings for child nodes in the AST.
    [req013] The get_ast_embeddings function shall return a list of the generated embeddings.
"""
import json
import logging
import importlib
import numpy as np
from typing import List, Dict, Tuple, Any

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    A class used to generate vector embeddings for Python code and generated questions and instructions.
    Attributes:
        tokenizer (AutoTokenizer): The tokenizer of the pretrained model used to tokenize the text.
        model (AutoModel): The pretrained model used to generate the embeddings.
    Methods:
        get_embeddings(text: str) -> List[float]: Generates a vector embedding for a given piece of text.
    """
    def __init__(self) -> None:
        try:
             # read in the embedding model details
            with open('config.json') as config_file:  # corrected file name
                self.config = json.load(config_file)[0]['embedding_model']
            
            # define the tokenizer and the model
            tokenizer_module_name, tokenizer_class_name = self.config['tokenizer_import_path'].rsplit('.', 1)
            model_module_name, model_class_name = self.config['model_import_path'].rsplit('.', 1)
            tokenizer_module = importlib.import_module(tokenizer_module_name)
            model_module = importlib.import_module(model_module_name)
            TokenizerClass = getattr(tokenizer_module, tokenizer_class_name)
            ModelClass = getattr(model_module, model_class_name)

            self.tokenizer = TokenizerClass.from_pretrained(self.config["tokenizer"])
            self.model = ModelClass.from_pretrained(self.config["model_path"])

        except (FileNotFoundError, json.JSONDecodeError, ImportError, AttributeError) as e:
            logger.error(f'Failed to load configuration file: {e}')
            self.tokenizer = None
            self.model = None

    def get_embeddings(self, text: str) -> List[List[float]]:
        """
        Generates a vector embedding for a given piece of text.
        Args:
            text (str): The text to generate an embedding for.
        Returns:
            List[List[float]]: The generated vector embeddings.
        """
        if not self.tokenizer or not self.model:
            logger.error('Embedding model not available.')
            return []

        # Tokenize the text into individual tokens
        tokens = self.tokenizer.tokenize(text)

        # Break the tokens into chunks if the total number of tokens exceeds max_length
        chunks = [tokens[i:i+self.config["max_seq_length"]] for i in range(0, len(tokens), self.config["max_seq_length"])]

        # Generate embeddings for each chunk
        embeddings = []
        for chunk in chunks:
            # Convert tokens back to text for this chunk
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk)
            inputs = self.tokenizer(chunk_text, return_tensors='pt', truncation=True, padding=True, max_length=self.config["max_seq_length"])
            outputs = self.model(**inputs)
            chunk_embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy().tolist()
            embeddings.append(chunk_embeddings)

        return embeddings


def get_ast_embeddings(ast: Dict[str, Any], generator: EmbeddingGenerator) -> List[List[float]]:
    """
    Generates embeddings for an abstract syntax tree (AST).
    Args:
        ast (Dict): The AST to generate embeddings for.
        generator (EmbeddingGenerator): The generator used to create the embeddings.
    Returns:
        List[List[float]]: The embeddings for the AST.
    """
    # Embed the node type
    embeddings = [generator.get_embeddings(ast['node_type'])]

    # If the node has a name, embed the name
    if 'name' in ast:
        embeddings.append(generator.get_embeddings(ast['name']))

    # If the node has children, recursively generate embeddings for the children
    if 'children' in ast:
        for child in ast['children']:
            embeddings.extend(get_ast_embeddings(child, generator))

    return embeddings


def simplify_ast(node):
    node_type = type(node).__name__
    
    if isinstance(node, ast.AST):
        fields = []
        for field in node._fields:
            value = getattr(node, field)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        fields.append(simplify_ast(item))
            elif isinstance(value, ast.AST):
                fields.append(simplify_ast(value))
        
        return (node_type, fields)
    
    else:
        return node_type

def get_vector_embeddings(base_name: str, file_details: Dict[str, Any], qa_list: List[Dict[str, Any]], instruct_list: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """
    Converts file details, QA pairs, and instructions into vector embeddings.
    Args:
        base_name (str): The base name of the file.
        file_details (Dict): A dictionary containing details of the Python file.
        qa_list (List[Dict]): A list of question-answer pairs.
        instruct_list (List[Dict]): A list of instructions.
    Returns:
        Tuple[List[List[float]], List[List[float]], List[List[float]]]: Three lists of vector embeddings, one for each input.
    """
    generator = EmbeddingGenerator()
    file_detail_embeddings = [generator.get_embeddings(base_name)]
    ast_heirarchy_embeddings = get_ast_embeddings(simplify_ast(file_details['file_info']['file_ast']), generator)
    file_detail_embeddings.append(ast_heirarchy_embeddings)
    ast_embeddings = get_ast_embeddings(file_details['file_info']['file_ast'], generator)
    file_detail_embeddings.append(ast_embeddings)

    qa_list_embeddings = [generator.get_embeddings(base_name+'.qa.json')]
    qa_embeddings = [generator.get_embeddings(json.dumps(qa)) for qa in qa_list]
    qa_list_embeddings.append(qa_embeddings)

    instruct_list_embeddings = [generator.get_embeddings(base_name+'.instruct.json')]
    instruct_embeddings = [generator.get_embeddings(json.dumps(instruct)) for instruct in instruct_list]
    instruct_list_embeddings.append(instruct_embeddings)
    
    return file_detail_embeddings, qa_list_embeddings, instruct_list_embeddings
