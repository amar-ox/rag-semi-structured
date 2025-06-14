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


# fastrag/script_generator.py

import importlib
import re
from jsonschema import validate
from langchain.schema import HumanMessage
from fastrag.text.utils import remove_python_code_fence


def import_libs(extractor_code):
    combined_namespace = {}
    import_pattern = r"(?:import (\w+))|(?:from (\w+) import (\w+)(?: as (\w+))?)"

    matches = re.findall(import_pattern, extractor_code)
    for full_import, from_import, import_as, alias in matches:
        if full_import:
            module = importlib.import_module(full_import)
            combined_namespace[full_import] = module
        elif from_import:
            module = importlib.import_module(from_import)
            if alias:
                combined_namespace[alias] = getattr(module, import_as)
            else:
                combined_namespace[import_as] = getattr(module, import_as)

    return combined_namespace

def process_sample_chunks(llm, prompt, sample_chunks, section=None):
    _stats = {
        "total_requests": 0,
        "total_in_chars": 0,
        "total_out_chars": 0,
        "total_time": 0
    }

    final_extractor = None
    final_code = ''

    for chunk in sample_chunks:
        retries = 5
        retry = 0
        extractor_code = ''
        error_message = ''
        success = False

        while retry < retries:
            if retry > 0:
                prompt_content = prompt.generate_prompt(input=chunk, section=section)
                feedback = prompt.generate_fb_prompt(extractor_code, error_message)
                messages = [
                    HumanMessage(content=prompt_content + '\n\n' + feedback),
                ]
            else:
                prompt_content = prompt.generate_prompt(input=chunk, section=section, code=final_code)
                messages = [
                    HumanMessage(content=prompt_content),
                ]

            # print(messages)
            output, execution_time = llm.invoke(messages)
            extractor_code = remove_python_code_fence(output)
            #print(f"Generated code: {extractor_code}")

            # Metrics
            _stats["total_requests"] += 1
            _stats["total_in_chars"] += sum(len(message.content) for message in messages)
            _stats["total_out_chars"] += len(output)
            _stats["total_time"] += execution_time

            try:
                # Import modules
                combined_namespace = import_libs(extractor_code)

                # Attempt to execute extractor
                exec(extractor_code, combined_namespace, combined_namespace)
                extract = combined_namespace['extract']
                json_result = extract(chunk)
                validate(instance=json_result, schema=prompt.current_schema(section))

                final_extractor = extract
                final_code = extractor_code
                success = True
                break  # Exit the retry loop on success
            except Exception as e:
                if hasattr(e, 'json_path'):
                    print(f"  In {e.json_path}, Object: {e.message}")
                    error_message = f"In {e.json_path}, Object: {e.message}"
                else:
                    print(f"  {e}")
                    error_message = str(e)
                retry += 1

        if success:
            continue  # Proceed to the next chunk
    _stats["average_latency"] = _stats["total_time"] / _stats["total_requests"]
    return final_extractor, final_code, _stats


def process_sample_chunks_per_section(llm, prompt, section_sample_chunks):
    section_extractors = {}
    section_extractors_code = {}

    # Initialize the overall stats object
    s2_stats = {
        'total_requests': 0,
        'total_in_chars': 0,
        'total_out_chars': 0,
        'total_time': 0.0,
        'average_latency': 0.0
    }

    for section, sample_chunks in section_sample_chunks.items():
        if section not in prompt.step2_schema:
            print(f" Skip section {section}: no schema provided")
            continue

        print(f" Script learning for section: {section}")
        extractor, extractor_code, _stats = process_sample_chunks(llm, prompt, sample_chunks, section)

        s2_stats['total_requests'] += _stats['total_requests']
        s2_stats['total_in_chars'] += _stats['total_in_chars']
        s2_stats['total_out_chars'] += _stats['total_out_chars']
        s2_stats['total_time'] += _stats['total_time']

        if extractor is not None:
            section_extractors[section] = extractor
            section_extractors_code[section] = extractor_code
            print(f"  Generated extractor for section: {section}")
        else:
            print(f"  Failed to generate extractor for section: {section}")
            break

    s2_stats["average_latency"] = s2_stats["total_time"] / s2_stats["total_requests"]
    return section_extractors, section_extractors_code, s2_stats

