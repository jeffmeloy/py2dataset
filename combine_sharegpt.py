import json
import os
import sys
import re
import random
import numpy as np
import math

from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif


def mean_pooling(model_output, attention_mask):
    """
    1.  Define token embeddings as the first element of the model output, only consider the first element 
        of the output because the model returns these additional elements: pooler_output, hidden_states, attentions.
        This step is necessary to extract the relevant token embeddings from the model's output.
    2.  Define the input mask expanded to represent the attention mask for each token, which zeroes out 
        embeddings for padded tokens, which is needed because the model returns embeddings for all tokens including 
        the padded tokens due to the max_length parameter that works by padding the input to the max_length.
        Handles variable-length sequences and ensure that padded tokens do not contribute to the pooling.
    3.  Multiply the token embeddings with the attention mask to zero out embeddings for padded tokens.
        Masks out the embeddings of padded tokens, preventing them from influencing the pooled representation.
    4.  Sum the embeddings for each token.
        Aggregates the token embeddings across the sequence dimension.
    5.  Divide the sum by the sum of the attention mask values for each token.
        Normalizes the pooled representation by the actual number of non-padded tokens, which is necessary to 
        account for the padding tokens that have been zeroed out to ensure that the pooled representation 
        is not biased by the number of padded tokens.
    7.  If the sum of the attention mask values is zero, we clamp the value at 1e-9 to avoid division by zero to
        handle the edge case where all tokens are padded, preventing numerical instability.
    8.  The resulting tensor will have shape (batch_size, embedding_dim) to rovide the final pooled representation 
        of the token embeddings for each input sequence.
    """
    token_embeddings = model_output[0]
    # define input_mask expanded to represent the attention mask for each token
    # which in needed to zero out embeddings for padded tokens
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )

    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def truncate_answer_if_needed(question, answer, max_length, tokenizer):
    """
    Truncate the answer if the total length of the question and answer exceeds the maximum length.
    1.  Return the question and answer as is if the total length is less than max_length, is quick check 
        to avoid unnecessary tokenization and truncation if the total length is already within the limit.
    2.  Tokenize the question and answer using the tokenizer, to convert the text into tokens that can 
        be counted and truncated.
    3.  Compute the total length of the tokens, to determine if the total length of the question and answer 
        tokens exceeds the maximum length.
    4.  If the total length is less than max_length, return the question and answer as is to avoid truncation 
        if the total length is already within the limit.
    5.  Otherwise, truncate the answer by the excess length, to ensure that the combined length of the question 
        and answer tokens fits within the maximum token length of the model.
    6.  Calculate the excess length is the total length minus max_length to determine the number of tokens 
        that need to be removed from the answer to fit within the maximum length.
    7.  Truncate the answer by removing the excess tokens from the end to preserve the question and 
        beginning of the answer.**need to ensure the dataset questions are shorter than the model's max_length.
    8.  Convert the tokens back to strings and return the truncated question and answer.
    """
    # quick check since length < token length
    if len(answer) + len(question) < max_length:
        return question, answer

    # get the tokens and return qestion if don't need to truncate
    tokens_question = tokenizer.tokenize(question)
    tokens_answer = tokenizer.tokenize(answer)
    total_length = len(tokens_question) + len(tokens_answer)
    if total_length < max_length:
        return question, answer

    # Truncate the answer by the excess length
    excess_length = total_length - max_length
    truncated_answer_length = len(tokens_answer) - excess_length
    tokens_answer = tokens_answer[: max(0, truncated_answer_length)]

    # convert the tokens back to strings and return
    question = tokenizer.convert_tokens_to_string(tokens_question)
    answer = tokenizer.convert_tokens_to_string(tokens_answer)
    return question, answer


def similiarity_filter(
    data: list[dict[str, any]],
    filter_ratio: float,
    max_length: int,
    model_max_length: int,
    similarity_threshold: float = 0.95,
):
    """
    Filter the dataset based on similarity scores and mutual information between question-answer pairs.
    1. Load the pre-trained BERT tokenizer and model.
    2. Separate the questions and answers from the conversation data to prepare them for embedding generation.
    3. Concatenate question and answer parts using a separator token.
    4. Convert the concatenated question-answer string into tokens that can be input to the BERT model.
    5. Generate embeddings for the question and answer using the model.
       This step generates dense vector representations (embeddings) of the question-answer pairs using the BERT model.
    6. Compute the mean pooling of token embeddings for question and answer.
       This step applies mean pooling to the token embeddings to obtain a single vector representation for each question-answer pair.
    7. Concatenate the question and answer embeddings along axis 0.
       This step combines the question and answer embeddings into a single vector for similarity calculations.
    8. Compute the cosine similarity matrix between all question-answer pairs.
       This step calculates the cosine similarity between each pair of question-answer embeddings to measure their similarity.
    9. Compute the sum of cosine similarity scores for each question-answer pair.
       This step calculates the sum of cosine similarity scores for each question-answer pair to assess its overall similarity to other pairs.
    10. Perform deduplication based on the similarity threshold.
        This step removes duplicate or highly similar question-answer pairs based on the specified similarity threshold.
    11. Adjust the filter ratio based on the number of selected indices.
        This step modifies the filter ratio to account for the reduction in dataset size after deduplication.
    12. Filter the deduplicated question-answer pairs based on the cosine similarity score.
        This step selects a subset of the deduplicated question-answer pairs based on their cosine similarity scores.
    13. Filter the data set further based on the mutual information between the question and answer in each pair.
        This step calculates the mutual information between the question and answer in each pair and selects a subset based on the mutual information scores.
    14. Return the filtered data set.
        This step returns the filtered dataset containing the selected question-answer pairs based on similarity and mutual information.
    """
    # Load the pre-trained BERT tokenizer and local model
    max_length = min(max_length, model_max_length)
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased", model_max_length=max_length
    )
    model = AutoModel.from_pretrained(
        ".\model", trust_remote_code=True, rotary_scaling_factor=2, device_map="auto"
    )
    model.eval()

    # Extract questions and answers from the dataset
    qa_pairs = []
    for entry in data:
        question_parts = []
        answer_parts = []
        for item in entry["conversation"]:
            if item.get("from") == "human":
                question_parts.append(item["value"])
            elif item.get("from") == "gpt":
                answer_parts.append(item["value"])

        # Concatenate question and answer parts using a separator token
        question = " [SEP] ".join(question_parts)
        answer = " [SEP] ".join(answer_parts)

        # Append the question-answer pair to the list if both are non-empty
        if question and answer:
            qa_pairs.append((question, answer))

    print(f"Number of question-answer pairs: {len(qa_pairs)}")
    print(f"Filter ratio: {filter_ratio}")

    qa_pair_embeddings = []
    count = 0

    # Set the device (GPU or CPU) for running the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Iterate over each question-answer pair
    for question, answer in qa_pairs:
        # Truncate the question and answer if needed to fit the max_length
        question, answer = truncate_answer_if_needed(
            question, answer, max_length, tokenizer
        )

        # Tokenize the question and answer
        question_input = tokenizer(
            question, padding=True, truncation=True, return_tensors="pt"
        )
        answer_input = tokenizer(
            answer, padding=True, truncation=True, return_tensors="pt"
        )

        # Move the input tensors to the device
        question_input = {k: v.to(device) for k, v in question_input.items()}
        answer_input = {k: v.to(device) for k, v in answer_input.items()}

        # Generate embeddings for the question and answer using the model
        with torch.no_grad():
            question_output = model(**question_input)
            answer_output = model(**answer_input)

        # Compute the mean pooling of token embeddings for question and answer
        q_emb = mean_pooling(question_output, question_input["attention_mask"])
        a_emb = mean_pooling(answer_output, answer_input["attention_mask"])

        # Convert the embeddings from PyTorch tensors to NumPy arrays
        q_emb = q_emb.cpu().numpy()
        a_emb = a_emb.cpu().numpy()

        # Concatenate the question and answer embeddings along axis 0
        qa_pair_embeddings.append(np.concatenate((q_emb, a_emb), axis=0))

        count += 1
        sys.stdout.write(f"\rProcessing pair {count} of {len(qa_pairs)}")

    # Convert the list of concatenated question-answer embeddings to a 2D NumPy array
    qa_pair_embeddings = np.array(qa_pair_embeddings)
    print(
        f"\nqa_pair_embeddings: {qa_pair_embeddings.shape}, {qa_pair_embeddings.dtype}"
    )

    # Reshape qa_pair_embeddings to have shape (num_pairs, 2 * embedding_dim)
    qa_pair_embeddings = qa_pair_embeddings.reshape(qa_pair_embeddings.shape[0], -1)

    # Compute the cosine similarity matrix between all question-answer pairs
    similarity_matrix = cosine_similarity(qa_pair_embeddings)
    print(f"similarity_matrix: {similarity_matrix.shape}, {similarity_matrix.dtype}")

    # Compute the sum of cosine similarity scores for each question-answer pair
    cosine_scores = (
        np.sum(similarity_matrix, axis=1) - 1
    )  # Subtract 1 to exclude self-similarity

    # Perform deduplication based on the similarity threshold
    selected_indices = []
    remaining_indices = list(range(len(qa_pair_embeddings)))

    while remaining_indices:
        i = remaining_indices[0]
        selected_indices.append(i)
        remaining_indices = [
            j
            for j in remaining_indices
            if similarity_matrix[i][j] <= similarity_threshold
        ]
    print(f"Duplicate elements removed: {len(qa_pairs) - len(selected_indices)}")

    # Filter the deduplicated question-answer pairs based on the cosine similarity score and mutual information between the question and answer
    deduplicated_qa_pairs = [qa_pairs[i] for i in selected_indices]
    deduplicated_cosine_scores = [cosine_scores[i] for i in selected_indices]
    filter_ratio = filter_ratio * (len(qa_pairs) / len(selected_indices))

    if filter_ratio >= 1:  # no filtering needed
        filtered_data = [data[i] for i in selected_indices]
    else:
        # First, filter based on cosine similarity score
        num_pairs = len(deduplicated_qa_pairs)
        num_selected_cosine = int(num_pairs * min(math.sqrt(filter_ratio), 1))
        sorted_cosine_indices = sorted(
            range(num_pairs), key=lambda i: deduplicated_cosine_scores[i]
        )
        selected_cosine_indices = sorted_cosine_indices[:num_selected_cosine]
        print(
            f"number of selected elements based on cosine similarity: {len(selected_cosine_indices)}"
        )

        # Compute mutual information scores for the selected question-answer pairs
        vectorizer = CountVectorizer()
        all_qa_texts = [f"{q} {a}" for q, a in qa_pairs]
        vectorizer.fit(all_qa_texts)  # Fit the vectorizer with the entire corpus

        selected_qa_pairs = [deduplicated_qa_pairs[i] for i in selected_cosine_indices]
        selected_qa_texts = [f"{q} {a}" for q, a in selected_qa_pairs]
        selected_qa_vectors = vectorizer.transform(selected_qa_texts)
        mutual_info_scores = mutual_info_classif(
            selected_qa_vectors, range(len(selected_qa_pairs))
        )

        # Filter based on mutual information score
        num_pairs_mi = len(selected_cosine_indices)
        num_selected_mi = int(num_pairs_mi * min(math.sqrt(filter_ratio), 1))
        sorted_mutual_info_indices = sorted(
            range(num_pairs_mi), key=lambda i: mutual_info_scores[i], reverse=True
        )
        selected_mutual_info_indices = sorted_mutual_info_indices[:num_selected_mi]
        print(
            f"number of selected elements based on mutual information: {len(selected_mutual_info_indices)}"
        )

        # Get the final selected indices
        final_selected_indices = [
            selected_cosine_indices[i] for i in selected_mutual_info_indices
        ]

        # Filter the data based on the final selected indices
        filtered_data = [data[i] for i in final_selected_indices]

    return filtered_data


def load_json_files(directory: str) -> list[dict[str, any]]:
    """
    1. List all the JSON files in the specified directory.
    2. Load each JSON file and preprocess the conversation data.
    3. Calculate the number of bytes for each conversation.
    4. Combine the data from all JSON files into a single list.
    5. Return the combined data.
    """
    json_files = [
        file
        for file in os.listdir(directory)
        if file.endswith(".json")
        and file not in ["train_sharegpt.json", "test_sharegpt.json", "algs.json"]
    ]

    combined_data = []
    for file in json_files:
        print(f"Loading file: {file}")
        with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error loading file: {file}, JSON decoding error: {e}")
                continue

            for entry in data:
                entry["conversation"] = preprocess_conversation(
                    entry.get("conversation", [])
                )
                entry["nbytes"] = sum(
                    len(json.dumps(msg, ensure_ascii=False).encode("utf-8"))
                    for msg in entry["conversation"]
                )
            combined_data.extend(data)
    return combined_data


def preprocess_conversation(conversation):
    """
    1. Insert an empty system message at the beginning of the conversation if it does not exist.
    2. Modify the system message to have an empty value.
    3. Return the modified conversation.
    """
    if not any(item.get("from") == "system" for item in conversation):
        conversation.insert(0, {"from": "system", "value": ""})
    return [modify_system_message(item) for item in conversation]


def modify_system_message(item):
    """
    1. Modify the system message to have an empty value.
    2. Return the modified item.
    """
    if item.get("from") == "system":
        item["value"] = ""
    return item


def remove_duplicates(data: list[dict[str, any]]) -> list[dict[str, any]]:
    """
    1. Compile a regex pattern to match AI-related terms.
    2. Iterate over each entry in the data.
    3. Filter out conversation items containing AI-related terms.
    4. Remove entries that do not have both human and GPT responses.
    5. Strip out the "Of course!" and "Certainly!" from the GPT response.
    6. Convert the conversation to a tuple of tuples.
    7. Deduplicate the data based on the unique conversation tuples.
    8. Return the deduplicated data.
    """
    # Compile the regex pattern outside of the loop for better performance
    ai_pattern = re.compile(
        r"(?:AI|language model|openai|I am sorry|I'm sorry|I apologize)"
    )

    unique_conversations = set()
    deduplicated_data = []

    for entry in data:
        filtered_conversation = [
            item
            for item in entry["conversation"]
            if not any(ai_pattern.search(v) for v in item.values())
        ]

        entry["conversation"] = filtered_conversation
        if not any(
            item.get("from") == "human" for item in filtered_conversation
        ) or not any(item.get("from") == "gpt" for item in filtered_conversation):
            continue

        # strip out the "Of course!" and "Certainly!" from gpt response
        if filtered_conversation[1].get("from") == "gpt":
            filtered_conversation[1]["value"] = (
                filtered_conversation[1]["value"]
                .replace("Of course! ", "")
                .replace("Certainly! ", "")
            )
        conversation_tuple = tuple(
            tuple(item.items()) for item in filtered_conversation
        )

        if conversation_tuple not in unique_conversations:
            unique_conversations.add(conversation_tuple)
            deduplicated_data.append(entry)

    return deduplicated_data


def save_json_files(
    data: list[dict[str, any]],
    output_file_json: str,
    output_file_jsonl: str,
    max_jsonl_size: int,
) -> None:
    """
    1. Sort the data based on the number of bytes in each conversation.
    2. Save the sorted data to a JSON file.
    3. Save the sorted data to a JSONL file with a maximum size limit.
    """
    sorted_data = sorted(data, key=lambda x: x["nbytes"])
    with open(output_file_json, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, indent=4)
    with open(output_file_jsonl, "w", encoding="utf-8") as f:
        for entry in sorted_data:
            jsonl_entry = {"chat": entry["conversation"]}
            entry_bytes = entry["nbytes"]
            if entry_bytes <= max_jsonl_size:
                json.dump(jsonl_entry, f)
                f.write("\n")
            else:
                break


def normalize_histogram(
    data: list[dict[str, any]],
    nbins: int,
    scaled: bool,
    use_similarity_filter: bool,
    model_max_length: int,
) -> list[dict[str, any]]:
    """
    1. Extract the number of bytes for each conversation.
    2. Compute the minimum and maximum number of bytes.
    3. Generate bin edges based on the minimum and maximum number of bytes.
    4. Compute the bin indices for each conversation based on the bin edges.
    5. Compute the bin counts for each bin.
    6. Compute the average count per bin.
    7. Compute the minimum count per bin.
    8. Compute the bin rank based on the bin counts.
    9. Normalize the data based on the bin rank.
    10. Return the normalized data.
    """
    nbytes_array = np.array([entry["nbytes"] for entry in data])
    min_nbytes, max_nbytes = nbytes_array.min(), nbytes_array.max()
    bin_edges = np.linspace(min_nbytes, max_nbytes, nbins + 1)
    bin_indices = np.digitize(nbytes_array, bin_edges) - 1
    bin_counts = np.bincount(bin_indices, minlength=nbins)
    avg_count = bin_counts.sum() / np.count_nonzero(
        bin_counts
    )  # average non-zero bin count
    min_count = max(avg_count / nbins, min_nbytes)
    bin_rank = np.argsort(bin_counts)[::-1]

    print("Min nbytes:", min_nbytes)
    print("Max nbytes:", max_nbytes)
    print("Bin counts:", bin_counts)
    print("Bin rank:", bin_rank)

    normalized_data = []
    for i, bin_index in enumerate(bin_rank):
        bin_data = [
            entry for entry, index in zip(data, bin_indices) if index == bin_index
        ]
        if bin_counts[bin_index] > 2:
            if scaled:
                scaled_count = min_count * (nbins - i)
                max_bin_length = max(entry["nbytes"] for entry in bin_data)
                print(
                    f"Bin {bin_index}: {len(bin_data)} records, scaled count: {scaled_count}, min count: {min_count}, max bin length: {max_bin_length}"
                )

                if use_similarity_filter:
                    bin_data = similiarity_filter(
                        bin_data,
                        scaled_count / len(bin_data),
                        max_bin_length,
                        model_max_length,
                    )
                else:
                    bin_data = random.sample(
                        bin_data, min(len(bin_data), int(scaled_count))
                    )
            else:
                bin_data = random.sample(bin_data, int(min_count))

            normalized_data.extend(bin_data)

    return normalized_data


def plot_histogram(data: list[dict[str, any]], nbins: int) -> None:
    """
    1. Extract the number of bytes for each conversation.
    2. Compute the minimum and maximum number of bytes.
    3. Generate bin edges based on the minimum and maximum number of bytes.
    4. Compute the histogram for the number of bytes.
    5. Print the histogram with the number of records in each bin.
    """
    nbytes_array = np.array([entry["nbytes"] for entry in data])
    min_nbytes, max_nbytes = nbytes_array.min(), nbytes_array.max()
    bin_edges = np.linspace(min_nbytes, max_nbytes, nbins + 1)
    histogram, _ = np.histogram(nbytes_array, bins=bin_edges)

    print("Histogram:")
    max_count = histogram.max()
    scale_factor = 100 / max_count
    for i in range(nbins):
        count = histogram[i]
        bar = "*" * int(count * scale_factor)
        print(f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}: {bar}, {count} records")


def main() -> None:
    """
    1. Parse the command-line arguments.
    2. Load the JSON files from the current directory.
    3. Sort the data based on the number of bytes in each conversation.
    4. Remove duplicates from the sorted data.
    5. Normalize the histogram based on the number of bytes.
    6. Split the data into test and train sets based on the normalized histogram.
    7. Save the test and train data to JSON and JSONL files.
    8. Print the total number of input and output records.
    9. Plot the histogram of the output records.
    """
    if len(sys.argv) < 4:
        print(
            "Usage: python script.py <max_jsonl_size> <nbins> [--norm] [--scaled] [--use_similarity_filter]"
        )
        sys.exit(1)

    system_value = (
        "Use reasoning and memory to master input meaning and intent to synthesize \
relevant and innovative responses. Enhance response coherence, clarity, and effectiveness \
through continual analysis, self-evaluation, and refinement."
    )

    max_jsonl_size = int(sys.argv[1])
    nbins = int(sys.argv[2])
    normalize = "--norm" in sys.argv
    scaled = "--scaled" in sys.argv
    use_similarity_filter = "--use_similarity_filter" in sys.argv

    print(
        f"input params:, max_jsonl_size: {max_jsonl_size}, nbins: {nbins}, normalize: {normalize}, scaled: {scaled}, use_similarity_filter: {use_similarity_filter}"
    )

    current_directory = os.getcwd()
    combined_data = load_json_files(current_directory)
    total_input_records = len(combined_data)
    sorted_data = sorted(combined_data, key=lambda x: x["nbytes"])
    sorted_data = [entry for entry in sorted_data if entry["nbytes"] <= max_jsonl_size]
    deduplicated_data = remove_duplicates(sorted_data)

    if normalize:
        deduplicated_data = normalize_histogram(
            deduplicated_data,
            nbins,
            scaled,
            use_similarity_filter,
            model_max_length=8192,
        )

    # Keep all of the algs data for test and train output
    with open("algs.json", "r", encoding="utf-8") as f:
        algs_data = json.load(f)
    deduplicated_data.extend(algs_data)
    deduplicated_data = sorted(deduplicated_data, key=lambda x: x["nbytes"])
    deduplicated_data = [
        entry for entry in deduplicated_data if entry["nbytes"] <= max_jsonl_size
    ]
    for entry in deduplicated_data:
        for item in entry["conversation"]:
            if item.get("from") == "system":
                item["value"] = system_value

    # If the first "from" value after the "system" value is "gpt" and not "human", then remove the entry from deduplicated_data
    deduplicated_data = [
        entry
        for entry in deduplicated_data
        if entry["conversation"][1].get("from") == "human"
    ]

    test_data = []
    train_data = []
    for i in range(nbins):
        bin_data = [
            entry for entry in deduplicated_data if entry["nbytes"] <= max_jsonl_size
        ]
        bin_data = sorted(bin_data, key=lambda x: x["nbytes"])
        bin_data = [entry for entry in bin_data if entry["nbytes"] <= max_jsonl_size]
        bin_data = [
            entry
            for entry in bin_data
            if entry["nbytes"] >= i * (max_jsonl_size / nbins)
        ]
        bin_data = [
            entry
            for entry in bin_data
            if entry["nbytes"] < (i + 1) * (max_jsonl_size / nbins)
        ]
        random.shuffle(bin_data)
        test_data.extend(bin_data[: len(bin_data) // 10])
        train_data.extend(bin_data[len(bin_data) // 10 :])

    save_json_files(
        test_data, "test_sharegpt.json", "test_sharegpt.jsonl", max_jsonl_size
    )
    save_json_files(
        train_data, "train_sharegpt.json", "train_sharegpt.jsonl", max_jsonl_size
    )
    total_output_records = len(deduplicated_data)
    print(f"Total input records: {total_input_records}")
    print(f"Total output records: {total_output_records}")

    plot_histogram(deduplicated_data, nbins)


if __name__ == "__main__":
    main()
