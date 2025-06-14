#    FastRAG: Efficient Retrieval Augmented Generation for Semi-structured Data
#    Copyright (C) 2024–2025 Amar Abane
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program. If not, see <https://www.gnu.org/licenses/>.


# fastrag/utils.py

import nltk
from nltk.data import find
import json
import tiktoken
import re
from .json_splitter import RecursiveJsonSplitter


def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4o")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def remove_json_code_fence(input_string): 
    if input_string.startswith("```"):
        input_string = input_string[3:]
    if input_string.endswith("```"):
        input_string = input_string[:-3]
    if input_string.startswith("json"):
        input_string = input_string[4:]
    if input_string.startswith("json_str"):
        input_string = input_string[8:]
    json_str = input_string.replace('null', '""')
    return json_str

def remove_python_code_fence(input_string):
    if input_string.startswith("```"):
        input_string = input_string[3:]  # Remove the first 3 characters
    if input_string.endswith("```"):
        input_string = input_string[:-3] # Remove the last 3 characters
    if input_string.startswith("python"):
        input_string = input_string[6:]
    return input_string

def extract_schema(text):
    schema = None
    text = text.replace("`", '')
    schema_start = text.find("{")
    schema_end = text.rfind("}") + 1
    schema_text = text[schema_start:schema_end]

    try:
        schema_text = schema_text.replace("'", '"')
        schema = json.loads(schema_text)
    except json.JSONDecodeError:
        schema = None

    return schema

def json_to_line_by_line(json_data):
    def flatten(obj, parent_key='', sep='/'):
        items = []
        if not obj:
            items.append((parent_key, ""))
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    objsIn = json_data
    linesOut = []

    if not isinstance(json_data, list):
        objsIn = [json_data]

    for obj in objsIn:
        if not isinstance(obj, dict):
            obj = json.loads(obj)
        flattened = flatten(obj)
        line_by_line = [f"{k}={v}" for k, v in flattened.items()]
        linesOut.extend(line_by_line)
    return linesOut

def chunk_json(data, message, max_allowed_tokens):
    prompt_tokens = num_tokens_from_string(message)
    chunk_tokens = max_allowed_tokens - prompt_tokens
    splitter = RecursiveJsonSplitter(min_chunk_size=chunk_tokens, max_chunk_size=chunk_tokens)
    chunks = splitter.split_json(json_data=data)
    return chunks

def split_text_into_lines(text):
    if not text:
        return []

    # Splitting the text by "\n" and then adding it back to each line
    lines = text.split("\n")
    # Adding "\n" back to each line except the last one if it was empty
    return [line + "\n" for line in lines[:-1]] + [lines[-1] if text[-1] != "\n" else lines[-1] + "\n"]

# FIXME: lines_overlap = 0 produces error
def chunk_lines(data, message, max_allowed_tokens, lines_overlap):
    prompt_tokens = num_tokens_from_string(message)
    chunk_tokens = max_allowed_tokens - prompt_tokens
    
    print(f"Max chunk size: {chunk_tokens} tokens")

    lines = split_text_into_lines(data)
    combined_chunks = []
    current_chunk = []
    current_chunk_tokens = 0

    for line in lines:
        line_tokens = num_tokens_from_string(line)
        if current_chunk_tokens + line_tokens < chunk_tokens:
            current_chunk.append(line)
            current_chunk_tokens += line_tokens
        else:
            combined_chunks.append(''.join(current_chunk))
            # Prepare for next chunk with overlap, if n_overlap > 0
            current_chunk = current_chunk[-lines_overlap:] if len(current_chunk) > lines_overlap else current_chunk[:]
            current_chunk.append(line)
            # Recalculate tokens for the new chunk including overlap
            current_chunk_tokens = sum(num_tokens_from_string(l) for l in current_chunk)

    # Add the last chunk if it's not empty
    if current_chunk:
        combined_chunks.append(''.join(current_chunk))

    return combined_chunks

def split_markdown_sections(data):
    # Define a pattern to match Markdown headers (e.g., #, ##)
    header_pattern = re.compile(r'^(#{1,6} )', re.MULTILINE)
    
    # Split the content at each header, capturing the headers themselves
    chunks = re.split(header_pattern, data)[1:]  # First element is empty if file starts with a header
    
    combined_chunks = [''.join(chunks[i:i + 2]) for i in range(0, len(chunks), 2)]
    return combined_chunks

def ensure_nltk_resources():
    """
    Ensures that the specified NLTK resources are available. Downloads them if missing.
    """
    resources = ["tokenizers/punkt", "corpora/stopwords"]
    for resource in resources:
        try:
            # Try to find the resource in the nltk data path
            find(resource)
            print(f"NLTK resource already available: {resource}")
        except LookupError:
            # Resource is missing, download it
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource)
