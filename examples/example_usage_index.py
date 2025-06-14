# examples/example_usage_index.py

import os
import json
import time
from langchain_community.graphs import Neo4jGraph

from fastrag import LLM
from fastrag.text import sample_chunks, sample_chunks_per_section
from fastrag.querying import compute_step1_coverage, compute_step2_coverage, KnowledgeGraphCreator, FastRAG
from fastrag.indexing import (
    Prompt,
    Loader,
    generate_schema_from_sample_chunks,
    extract_steps_schemas,
    process_sample_chunks,
    process_sample_chunks_per_section,
    extract_data,
    extract_data_per_section,
)


s1_data_file_name = "step1_data_output.json"
s2_data_file_name = "step2_data_output.json"
s1_schema_file_name = "step1_schema.json"
s2_schema_file_name = "step2_schema.json"

# Set Up LLM
API_KEY = os.environ.get("API_KEY")
llm = LLM(model_name="gpt-4o", api_key=API_KEY, temperature=0.1)

#################### DATA EXTRACTION STEP 1:

# Step 2: Init prompt
prompt = Prompt()

# Step 2: Load Data
loader = Loader(prompt, max_allowed_tokens=1200)
data_path = "../data/cisco"
data_files = ["as1border1"]      # w/o extention
file_chunks, s1_num_chunks = loader.prepareDataset1(data_path, data_files)

# Step 3: Chunk Sampling
combined_chunks = []
for file, chunks in file_chunks.items():
    combined_chunks.extend(chunks)

# Step 4: Find learning parameters
N = 1       # Desired number of sample chunks
m_sample_chunks, m_eval_chunks = sample_chunks(combined_chunks, N, return_evals=True)
if not m_sample_chunks:
    print("Failed to find hyperparameters")
    exit(1)

# Step 5: Schema Learning
print(f"Learning schema...")
schema, s0_stats = generate_schema_from_sample_chunks(llm, prompt, m_sample_chunks)
if schema:
    print(f"Extracted schema: {json.dumps(schema)}")
else:
    print(f"**Stop: Couldn't extract a schema.")
    exit(1)

# separate schema intro step1 and step2
step1_schema, step2_schema = extract_steps_schemas(schema)
print(f"Step 1 schema {json.dumps(step1_schema)}")
print(f"Step 2 schema {json.dumps(step2_schema)}")

prompt.step1_schema = step1_schema
prompt.step2_schema = step2_schema

# save schemas
with open(s1_schema_file_name, 'w') as json_file:
    json.dump(step1_schema, json_file)
with open(s2_schema_file_name, 'w') as json_file:
    json.dump(step2_schema, json_file)


# Step 6: Script learning
print(f"Learning scripts...")
extractor, extractor_code, s1_stats = process_sample_chunks(llm, prompt, m_sample_chunks)
if extractor is not None:
    print(f"Generated extractor.")
else:
    print(f"**Stop: Couldn't generate an extractor.")
    exit(1)


# Step 7: Data Extraction
print(f"Processing data...")
step1_data_output = extract_data(file_chunks, extractor)
print(json.dumps(step1_data_output))

# save output
with open(s1_data_file_name, 'w') as json_file:
    json.dump(step1_data_output, json_file)

try:
    with open(s1_data_file_name, 'r') as file:
        step1_data_output = json.loads(file.read())

    with open(s1_schema_file_name, 'r') as file:
        prompt.step1_schema = json.loads(file.read())

    with open(s2_schema_file_name, 'r') as file:
        prompt.step2_schema = json.loads(file.read())
except FileNotFoundError:
    print(f"File not found")

# Step 8: Compute coverage
step1_cov = compute_step1_coverage(file_chunks, step1_data_output, print_missing=False)


#################### DATA EXTRACTION STEP 2:
# Step 9: Chunk section data for Step 2
section_combined_chunks, s2_num_chunks = loader.prepareDataset2(step1_data_output)

# Step 10: Find learning parameters for each section
NS = 1       # Desired number of sample chunks
section_sample_chunks = sample_chunks_per_section(section_combined_chunks, NS)
if len(section_sample_chunks.items()) != len(section_combined_chunks.items()):
    print("Failed to find hyperparameters")
    exit(1)

# Step 11: Script learning for each section
print(f"Learning scripts...")
section_extractors, section_extractors_code, s2_stats = process_sample_chunks_per_section(llm, prompt, section_sample_chunks)
if len(section_extractors.items()) == len(section_sample_chunks.items()):
    print(f"Generated all section extractors.")
else:
    print(f"**Stop: Couldn't generate all section extractor.")
    exit(1)

# Step 12: Data extraction for each section
print(f"Processing data...")
final_results = extract_data_per_section(step1_data_output, section_extractors)
print(json.dumps(final_results))

# save output
with open(s2_data_file_name, 'w') as json_file:
    json.dump(final_results, json_file)
    
# Step 8: Compute coverage
step1_cov = compute_step2_coverage(file_chunks, final_results, print_missing=False)


#################### Retrieval:
# Read final_result file
try:
    with open(s2_data_file_name, 'r') as file:
        final_results = json.loads(file.read())
except FileNotFoundError:
    print(f"File not found: {s2_data_file_name}")
# print(json.dumps(final_results))

# Create the Knowledge Graph
print(f"Creating Knowledge Graph...")
graph_creator = KnowledgeGraphCreator(uri="bolt://localhost:7687", user="neo4j", password="12345-password")
start_time = time.time()
graph_creator.create_graph(final_results)
end_time = time.time()
graph_creator.close()
print(f"Duration: {end_time - start_time} seconds")

# close connection to database
fast_rag.close()


# generate example questions from non-sampled chunks:
print(f"Generating example questions...")
if m_eval_chunks:
    example_questions = FastRAG.generate_questions(llm, m_eval_chunks)
    print(example_questions)
