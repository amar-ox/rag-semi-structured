# FastRAG: Efficiently Processing Semi-Structured Network Data with LLMs

## Introduction

FastRAG (Fast Retrieval-Augmented Generation) is a system designed to efficiently process large volumes of semi-structured network data, such as logs and configurations, using Large Language Models (LLMs). By combining schema and script learning with smart chunk sampling, FastRAG minimizes reliance on LLMs, reducing both time and cost.

This repository contains the code and instructions to reproduce the results presented in our [Article](https://arxiv.org/abs/2411.13773).


## Features

- **Efficient Data Processing**: Processes semi-structured data without sending all chunks through an LLM.
- **Schema and Script Learning**: Generates JSON schemas and Python parsing scripts using LLMs.
- **Knowledge Graph Creation**: Constructs a Neo4j Knowledge Graph for efficient querying.
- **Multiple Querying Methods**: Supports graph querying, text search, combined querying, and hybrid querying.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Operating System**: Linux, macOS, or Windows
- **Python**: Version 3.7 or higher
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

Open your web browser and navigate to [http://localhost:7474](http://localhost:7474). 
Log in with the username `neo4j` and the password `12345-password`.
Here you can later see the Knowledge Graph created by FastRAG after processing the dataset.


## Example Usage

The `examples/` folder contains scripts that demonstrate the full FastRAG pipeline on an example dataset.

### Step 1: Set your OpenAI API key

Before running any example, make sure your LLM API key is available as an environment variable:

```bash
export API_KEY=...
```

### Step 2: Build the Knowledge Graph

Run the following script to ingest data and construct the knowledge graph:

```bash
cd examples
python example_usage_index.py
```

### Step 3: Query the RAG system

Once the graph is built, you can ask questions using the querying script:

```bash
python example_usage_query.py -q "how many devices are in my network?" -r text
```

The `-r` option specifies the retrieval mode and accepts one of the following values:

* `graph` – Query using graph-based retrieval.
* `text` – Query using text-based retrieval.
* `hybrid` – Combine graph and text retrieval.
* `combined` – Fuse all retrieval results before generation.

> See the accompanying [FastRAG Article](https://arxiv.org/abs/2411.13773) for a detailed explanation of each retrieval type.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements, bug fixes, or new features.