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


import os
import json
import math
from collections import Counter

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import word_tokenize

from fastrag.text.utils import chunk_lines


class Loader:
    """
    A class to load files.
    """

    def __init__(self, prompt, max_allowed_tokens=2049):
        """
        Initializes the Loader with a path to the files, the prompt object, and max input tokens in a prompt.

        :param path: str - The path where files are located.
        :param prompt: Prompt - A Prompt object containing the prompt template.
        :param max_allowed_tokens: number - Max tokens in each chunk.
        """
        self._max_allowed_tokens = max_allowed_tokens
        self._prompt = prompt

    
    def prepareDataset1(self, path, files):
        extensions = ('.cfg', '.txt', '.json', '.log')

        file_chunks = {}   # for processing: files content split into chunks
        total_chunks = 0
    
        # Get list of all .cfg files
        all_files = [f for f in os.listdir(path) if f.endswith(extensions)]

        # If files is empty, process all files
        if not files:
            files_to_process = all_files
        else:
            # Otherwise, process only the ones defined in files
            files_to_process = [f"{file_name}{ext}" for file_name in files for ext in extensions if f"{file_name}{ext}" in all_files]

        for file_name in files_to_process:
            file_path = os.path.join(path, file_name)

            # Get the file name without the extension
            file_name = os.path.splitext(file_name)[0]

            with open(file_path, 'r') as file:
                text = file.read()

                # compute chunks:
                message = self._prompt.generate_prompt(input="")
                chunks = chunk_lines(text, message, self._max_allowed_tokens, 5)
                file_chunks[file_name] = chunks
                total_chunks+=len(chunks)

        print(f"Processed {len(files_to_process)} file(s) for a total chunks of {total_chunks}.")
        return file_chunks, total_chunks
    
    
    def prepareDataset2(self, step1_results):
        file_section_chunks = {}
        total_chunks = 0
        for file, section_data in step1_results.items():
            file_section_chunks[file], num_chunks = self._chunk_file_sections(section_data)
            total_chunks+=num_chunks

        # combine all chunks per section (across files):
        section_combined_chunks = {}
        for file, section_chunks in file_section_chunks.items():
            for section, chunks in section_chunks.items():
                if section not in section_combined_chunks:   # new section
                    section_combined_chunks[section] = chunks
                else:
                    section_combined_chunks[section].extend(chunks)
                    
        return section_combined_chunks, total_chunks
    
    def _chunk_file_sections(self, section_data):
        section_chunks = {}   # return sections content split into chunks
        total_chunks = 0

        for section, data in section_data.items():
            if not data:
                print(f"    Skip section {section}: no data")
                section_chunks[section] = []
                continue

            # compute chunks:
            message = self._prompt.generate_prompt(input="", section=section)
            
            if message is None:   # using schema to control which section to not process
                print(f"    Skip section {section}: no schema provided")
                section_chunks[section] = [data]
                continue

            chunks = chunk_lines(data, message, self._max_allowed_tokens, 5)
            section_chunks[section] = chunks
            total_chunks+=len(chunks)
            
            print(f"    Chunked section {section}: {len(chunks)} chunks")

        print(f"  Total chunks: {total_chunks}")
        return section_chunks, total_chunks
