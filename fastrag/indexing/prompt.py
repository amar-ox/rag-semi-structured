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


############## code generation step1
STEP1_PROMPT_TEMPLATE_FIRST = """You are a network engineer using Python. Your task is to analyze the network data provided below and write a Python function named `extract` that parses it and maps each input line into the correct object in the given JSON schema.

The function should:
- Take a string parameter (the network data).
- Return a dictionary formatted according to the provided JSON schema.

You will receive new syntaxes to process in the future, so ensure your function is robust and easy to update.

Only return the Python function -- not other text.

JSON Schema: {schema}

Input Data: {input}

Extraction function:"""


STEP1_PROMPT_TEMPLATE_ITER = """You are a network engineer using Python. You have already created the following function that maps each line from the input to the correct object in the given JSON schema:

```
{code}
```

Refine this function to correctly process the following data: {input}

IMPORTANT: You must look for new syntaxes in the input data that are not supported by the function given above.

The extracted data must be formatted according to the provided JSON schema: {schema}

Ensure that the function's signature and existing extraction capabilities are preserved. 
Only return the Python function -- not other text."""


############## code generation step2
STEP2_PROMPT_TEMPLATE_FIRST = """Your task is to write a Python function named `extract` that parses the given network data and extracts structured output according to the given JSON schema.

The function should:
- Take a string parameter (the network data).
- Parse the input data lines to extract and structure objects and their properties.
- Use 'input_data' property of each object to keep track of all the input lines that define that object.
- Return a dictionary formatted according to the provided JSON schema.

You will receive new syntaxes to process in the future, so ensure your function is simple and easy to maintain.

Return only the Python function --no other text.

JSON Schema: {schema}

Input Data: {input}

Extraction function:"""


STEP2_PROMPT_TEMPLATE_ITER = """You have already created the following function that parses network data and extracts structured output:

```
{code}
```

Refine this function to correctly process the following input data: {input}

IMPORTANT: You must look for new syntaxes in the input data that are not supported by the given function.

The function should:
- Take a string parameter (the network data).
- Parse the input data lines to extract and structure objects and their properties.
- Use 'input_data' property of each object to keep track of all the input lines that define that object.
- Return a dictionary formatted according to the provided JSON schema.

The extracted data must be formatted according to the provided JSON schema: {schema}

Return only the Python function --no other text."""


############## schema extraction templates w/o examples
SCHEMA_PROMPT_TEMPLATE_FIRST = """You are a data scientist. Your task is to analyze the provided network data and extract entity types.

-Steps-

1. Identify Entity Types and Properties:
  - Identify and list all relevant entity types from the data. 
  - Do not use generic entity types (e.g., log entry, configuration, etc.). Make sure the entity types are clearly distinguishable from one another.
  - For each identified entity type, identify and list its main properties. Extract only the most important properties.
  - TIP: It is better to extract more entity types than a few entity types with a lot of properties.

2. Construct JSON Schema:
  - Create a Python jsonschema that outlines in the 1st level each entity type and its properties. 
  - Keep the jsonschema coherent and concise.
  - Include a short description for each entity type in the schema.
  - IMPORTANT: Each entity type should be defined in an array to represent multiple instances.


Data
```
{input}
```

Return only the jsonschema --no other text."""


SCHEMA_PROMPT_TEMPLATE_ITER = """You are a data scientist. Your task is to analyze the provided network data, extract entity types and refine the given schema accordingly.

-Steps-

1. Identify Entity Types and Properties:
  - Identify and list all relevant entity types from the data.
  - Do not use generic entity types (e.g., log entry, configuration, etc.). Make sure the entity types are clearly distinguishable from one another.
  - For each identified entity type, identify and list its main properties. Extract only the most important properties.
  - TIP: It is better to extract more entity types than a few entity types with a lot of properties.

2. Refine JSON Schema:
  - Analyze the given JSON schema and refine it to include the new entity types and their properties.
  - Keep the jsonschema coherent, simple, and concise.
  - Include a short description for each new or refined entity type in the schema.
  - IMPORTANT: Each entity type should be defined in an array to represent multiple instances.


Given Schema:
```
{schema}
```

New Data:
```
{input}
```

Return only the jsonschema --no other text."""


FB_TEMPLATE = """\nYou already created this output in a previous attempt:

```
{invalid_output}
```

However, this doesn't comply with the above requirements and triggered this error: {error_message}.

Correct the function and try again. Just return the corrected output without any explanations."""


class Prompt:

    def __init__(self):
        self._step1_schema = {}
        self._step2_schema = {}
        
    @property
    def step1_schema(self):
        return self._step1_schema

    @step1_schema.setter
    def step1_schema(self, value):
        self._step1_schema = value

    @property
    def step2_schema(self):
        return self._step2_schema

    @step2_schema.setter
    def step2_schema(self, value):
        self._step2_schema = value

    def current_schema(self, section=None):
        if not section:
            return self._step1_schema
        else:
            return self._step2_schema[section]
        
    def generate_prompt(self, input, section=None, code=None):
        if not section:    # we are in step 1
            return self.generate_prompt1(input=input, code=code)
        return self.generate_prompt2(input=input, section=section, code=code)


    def generate_prompt1(self, input, code=None):
        if not code:
            return STEP1_PROMPT_TEMPLATE_FIRST.format(
                schema=self._step1_schema,
                input=input)
        else:
            return STEP1_PROMPT_TEMPLATE_ITER.format(
                schema=self._step1_schema,
                input=input,
                code=code)

    def generate_prompt2(self, input, section, code=None):
        if section not in self._step2_schema:
            return None
        if not code:
            return STEP2_PROMPT_TEMPLATE_FIRST.format(
                schema=self._step2_schema[section],
                input=input)
        else:
            return STEP2_PROMPT_TEMPLATE_ITER.format(
                schema=self._step2_schema[section],
                input=input,
                code=code)
            
    def generate_prompt_schema(self, input, schema=None):
        if not schema:
            return SCHEMA_PROMPT_TEMPLATE_FIRST.format(input=input)
        else:
            return SCHEMA_PROMPT_TEMPLATE_ITER.format(
                input=input,
                schema=schema)
        

    def generate_fb_prompt(self, invalid_output, error_message):
        return FB_TEMPLATE.format(
                invalid_output=invalid_output, 
                error_message=error_message)
