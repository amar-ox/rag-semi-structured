# FastRAG: Efficiently Processing Semi-Structured Network Data with LLMs

## Introduction

FastRAG (Fast Retrieval-Augmented Generation) is a system designed to efficiently process large volumes of semi-structured network data, such as logs and configurations, using Large Language Models (LLMs). By combining schema and script learning with smart chunk sampling, FastRAG minimizes reliance on LLMs, reducing both time and cost.

This repository contains the code and instructions to reproduce the results presented in our [Medium article](https://medium.com/your-article-link). You will learn how to set up the environment, run the code, and experiment with FastRAG on your own datasets.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Setting Up Neo4j](#setting-up-neo4j)
- [Usage](#usage)
  - [1. Launch Jupyter Notebook](#1-launch-jupyter-notebook)
  - [2. Set Up LLM API Key](#2-set-up-llm-api-key)
  - [3. Loading Data](#3-loading-data)
  - [4. Chunk Sampling](#4-chunk-sampling)
  - [5. Schema and Script Learning](#5-schema-and-script-learning)
  - [6. Creating the Knowledge Graph](#6-creating-the-knowledge-graph)
  - [7. Querying](#7-querying)
- [Reproducing the Results](#reproducing-the-results)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Efficient Data Processing**: Processes semi-structured data without sending all chunks through an LLM.
- **Schema and Script Learning**: Generates JSON schemas and Python parsing scripts using LLMs.
- **Knowledge Graph Creation**: Constructs a Neo4j Knowledge Graph for efficient querying.
- **Multiple Querying Methods**: Supports graph querying, text search, combined querying, and hybrid querying.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Operating System**: Linux, macOS, or Windows
- **Python**: Version 3.7 or higher
- **Jupyter Notebook**: Installed on your system
- **Docker**: Installed and running
- **LLM API Access**: An API key for an LLM service (e.g., OpenAI's GPT-4)
- **Neo4j**: Knowledge Graph database (to be set up via Docker)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/fastrag.git
   cd fastrag
   ```

2. **Install Python Dependencies**

   We recommend using a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

   Install the required packages:

   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## Setting Up Neo4j

FastRAG uses Neo4j as a Knowledge Graph database. Follow these steps to set up Neo4j using Docker.

1. **Pull the Neo4j Docker Image**

   ```bash
   docker pull neo4j:5.18
   ```

2. **Run the Neo4j Docker Container**

   ```bash
   docker run --name neo4j \
     -p 7687:7687 -p 7474:7474 \
     -v /tmp/neo4j/data:/data \
     -e NEO4J_AUTH=neo4j/12345-password \
     -e NEO4J_PLUGINS='["apoc","apoc-extended"]' \
     -e NEO4J_apoc_export_file_enabled=true \
     -e NEO4J_apoc_import_file_enabled=true \
     -e NEO4J_apoc_import_file_use__neo4j__config=true \
     -e NEO4J_dbms_security_procedures_unrestricted='apoc.*,algo.*,net.*' \
     -e NEO4J_dbms_security_procedures_allowlist='apoc.*,algo.*,net.*' \
     neo4j:latest
   ```

   **Note**: Ensure that the `/tmp/neo4j/data` directory exists on your system, or change the volume mapping to a directory of your choice.

3. **Verify Neo4j is Running**

   Open your web browser and navigate to [http://localhost:7474](http://localhost:7474). Log in with the username `neo4j` and the password `12345-password`.

## Usage

### 1. Launch Jupyter Notebook

Start Jupyter Notebook in the repository directory:

```bash
jupyter notebook
```

Open the provided notebook (e.g., `fastrag_notebook.ipynb`).

### 2. Set Up LLM API Key

In the notebook, set your LLM API key:

```python
import getpass

API_KEY = getpass.getpass('LLM API Key:')
```

Enter your API key when prompted.

### 3. Loading Data

Load your dataset (logs or configurations) into the notebook:

```python
from loader import Loader

prompt = "Your LLM prompt here"
MAX_ALLOWED_TOKENS = 2048  # Adjust based on your LLM's context window
data_path = "path/to/your/data"
data_files = ["file1.log", "file2.log"]  # List of files to process

loader = Loader(prompt, MAX_ALLOWED_TOKENS)
file_chunks = loader.prepareDataset1(data_path, data_files)
```

Replace `data_path` and `data_files` with the path to your dataset and the list of files to process.

### 4. Chunk Sampling

Adjust the parameters to select representative chunks:

```python
from chunk_sampler import find_parameters

N = 5  # Number of clusters or samples; adjust as needed
combined_chunks = sum(file_chunks.values(), [])  # Flatten the list of chunks
sample_chunks, eval_chunks = find_parameters(combined_chunks, N)
```

- **Adjusting Parameters**: Experiment with `N` (number of samples) to find the optimal sample size that covers the syntax of your dataset.

### 5. Schema and Script Learning

Generate the JSON schema and parsing scripts:

```python
from schema_generator import generate_schema_sample_chunks
from script_generator import process_sample_chunks

# Generate Schema
schema, s0_stats = generate_schema_sample_chunks(sample_chunks)

# Create Step 1 and Step 2 Schemas
step1_schema, step2_schema = create_step_schemas(schema)

# Generate Parsing Scripts
extractor, extractor_code, s1_stats = process_sample_chunks(sample_chunks)
```

- **Note**: The functions `generate_schema_sample_chunks` and `process_sample_chunks` will prompt the LLM to generate the schema and parsing code based on your sample chunks.

### 6. Creating the Knowledge Graph

Insert the parsed data into Neo4j:

```python
from graph_creator import Neo4jGraphCreator

# Initialize the Graph Creator
neo4j_graph_creator = Neo4jGraphCreator(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="12345-password"
)

# Parse the Entire Dataset
final_results = {}
for file, chunks in file_chunks.items():
    parsed_data = extractor("\n".join(chunks))
    final_results[file] = parsed_data

# Create the Knowledge Graph
neo4j_graph_creator.create_graph_from_dict(final_results)
```

- **Tip**: Ensure that the `uri`, `user`, and `password` match your Neo4j configuration.

### 7. Querying

Test different querying methods:

```python
from query_chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# Define Prompts
GRAPH_RAG_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template="Your graph RAG prompt here"
)
TEXT_RAG_PROMPT = PromptTemplate(
    input_variables=["question"], template="Your text RAG prompt here"
)
HYBRID_RAG_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template="Your hybrid RAG prompt here"
)

# Initialize Chains
graph_chain = GraphCypherQAChain.from_llm(
    llm_model, 
    graph=neo4j_graph_creator.graph,
    cypher_prompt=GRAPH_RAG_PROMPT,
)
text_chain = GraphCypherQAChain.from_llm(
    llm_model,
    graph=neo4j_graph_creator.graph,
    cypher_prompt=TEXT_RAG_PROMPT,
)
hybrid_chain = GraphCypherQAChain.from_llm(
    llm_model, 
    graph=neo4j_graph_creator.graph,
    cypher_prompt=HYBRID_RAG_PROMPT,
)

# Combined Querying
combined_chain = (
    {"text_rag_context": text_chain, "graph_rag_context": graph_chain, "question": RunnablePassthrough()}
    | your_combined_prompt
    | llm_model
    | StrOutputParser()
)

# Execute Queries
question = "Your question here"
graph_response = graph_chain.invoke(question)
text_response = text_chain.invoke(question)
hybrid_response = hybrid_chain.invoke(question)
combined_response = combined_chain.invoke(question)
```

- **Note**: Replace `llm_model` with your LLM model instance and provide appropriate prompts.

## Reproducing the Results

To reproduce the results from our article:

1. **Use the Provided Datasets**

   The datasets used in the article are included in the `datasets` directory.

2. **Follow the Notebook Steps**

   Execute each cell in the notebook sequentially, ensuring that you adjust any parameters as needed.

3. **Compare Results**

   Compare your outputs with the results shown in the article to verify accuracy.

4. **Adjust Parameters for Evaluation**

   Experiment with different sample sizes (`N`), `n_clusters`, and `n_top` values to see their effect on the coverage ratio and extraction performance.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements, bug fixes, or new features.

## License

This project is licensed under the [MIT License](LICENSE).


# Install package
`pip install -e .`