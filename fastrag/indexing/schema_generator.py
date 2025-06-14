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


# fastrag/schema_generator.py

import json
from jsonschema import Draft7Validator, exceptions
from langchain.schema import HumanMessage
from fastrag.text.utils import remove_json_code_fence, extract_schema


def generate_schema_from_sample_chunks(llm, prompt, sample_chunks):
    _stats = {
        "total_requests": 0,
        "total_in_chars": 0,
        "total_out_chars": 0,
        "total_time": 0
    }

    final_schema = ''

    for chunk in sample_chunks:
        retries = 3
        retry = 0
        schema = ''
        error_message = ''
        previous_schema = final_schema  # Use the schema from the previous iteration

        while retry < retries:
            if retry > 0:
                prompt_content = prompt.generate_prompt_schema(input=chunk)  # Use the first prompt
                feedback = prompt.generate_fb_prompt(schema, error_message)
                messages = [
                    HumanMessage(content=prompt_content + '\n\n' + feedback),
                ]
            else:
                prompt_content = prompt.generate_prompt_schema(  # Use first or iter prompt depending on schema
                    input=chunk,
                    schema=previous_schema
                )
                messages = [
                    HumanMessage(content=prompt_content),
                ]

            # print(messages)
            output, execution_time = llm.invoke(messages)
            # print(output)

            # Update metrics
            _stats["total_requests"] += 1
            _stats["total_in_chars"] += sum(len(message.content) for message in messages)
            _stats["total_out_chars"] += len(output)
            _stats["total_time"] += execution_time

            try:
                # Validate schema
                output = remove_json_code_fence(output)
                schema = extract_schema(output)
                # This will raise a ValidationError if the schema is not well-formed
                Draft7Validator.check_schema(schema)
                break  # Exit the retry loop if schema is valid
            except exceptions.SchemaError as e:
                print(f"  Schema validation error: {e}")
                error_message = str(e)
                retry += 1

        final_schema = schema  # Update the final schema with the latest schema
    _stats["average_latency"] = _stats["total_time"] / _stats["total_requests"]
    return final_schema, _stats


def extract_steps_schemas(input_schema):
    step1_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False
    }

    step2_schema = {}

    for prop, definition in input_schema["properties"].items():
        step1_schema["properties"][prop] = {
            "type": "string",
            "description": f"All input lines related to {definition['description']}"
        }
        step1_schema["required"].append(prop)

        if definition["type"] == "array":
            step2_schema[prop] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }

            for subprop, subdef in definition["items"]["properties"].items():
                step2_schema[prop]["items"]["properties"][subprop] = subdef
            
            step2_schema[prop]["items"]["properties"]["input_data"] = {
                "description": "All input lines covering this item.",
                "type": "string"
            }
            step2_schema[prop]["items"]["required"] = definition["items"].get("required", []) + ["input_data"]

        else:
            # Transform the level-1 object into an array
            step2_schema[prop] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }

            for subprop, subdef in definition["properties"].items():
                step2_schema[prop]["items"]["properties"][subprop] = subdef

            step2_schema[prop]["items"]["properties"]["input_data"] = {
                "description": "All input lines covering this item.",
                "type": "string"
            }
            step2_schema[prop]["items"]["required"] = definition.get("required", []) + ["input_data"]

    return step1_schema, step2_schema
