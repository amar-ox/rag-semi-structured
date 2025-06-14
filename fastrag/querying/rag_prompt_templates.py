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


from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

GRAPH_RAG_TEMPLATE = """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query that retrieves relevant answer.

Here is the schema information:
{schema}.

Use only the provided relationship types and properties in the schema. Do not use any other relationship types or properties that are not provided.

Respond with only Cypher statements. Do not answer with any explanations.

User input: {question}
Cypher query: """

GRAPH_RAG_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=GRAPH_RAG_TEMPLATE
)

###

TEXT_RAG_TEMPLATE = """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query that performs a fulltext search on the index `inputDataLines`.

Steps:
- Understand the question and define the search expression to use refering to the documentation given below. Remember to explore all the features in the documentation given. 
- Construct a syntactically correct Cypher using only the fulltext search on the `inputDataLines` index.
- Note that characters `:` and `/` and `-` should always be escaped with `\`.

Respond with the Cypher query only --no other text or explanations.


* Use the documentation below to know how to query the text search index:

- Specific attribute (complete match):

EXAMPLE_1:
- Input question: Search for movies that have the word “matrix” in their title.
- Cypher query: 
```CALL db.index.fulltext.queryNodes("inputDataLines", "matrix") YIELD node, score RETURN node, score```
END_OF_EXAMPLE_1


- Logical operator:
There are two logical operators available, “OR” and “AND”.

EXAMPLE_2:
- Input question: Search for movies that have the words "war" OR "matrix: revolution" in their title.
- Cypher query:
```CALL db.index.fulltext.queryNodes("inputDataLines", "'the' OR 'matrix\: revolution'") YIELD node, score RETURN node, score```
END_OF_EXAMPLE_2


- Single-character wildcard:
The single-character wildcard operator `?` looks for terms that match that with the single character replaced.

EXAMPLE_3:
- Input question: Can you show me movies whose title begins with 'ma'?
- Cypher query:
```CALL db.index.fulltext.queryNodes("inputDataLines", "ma?") YIELD node, score RETURN node, score```
END_OF_EXAMPLE_3


- Multi-character wildcard:
The multi-character wildcard `*` operator looks for zero or more characters. It can be used before, after, or in the middle of an expression.

EXAMPLE_4:
- Input question: What are movies with title containing 'sun'?
- Cypher query:
```CALL db.index.fulltext.queryNodes("inputDataLines", "*sun*") YIELD node, score RETURN node, score```
END_OF_EXAMPLE_4


- Fuzzy search:
Fuzzy search `~` works by using mathematical formulae that calculate the similarity between two words.

EXAMPLE_5:
- Input question: Can you find movies whose titles are similar to "matrix"?
- Cypher query:
```CALL db.index.fulltext.queryNodes("inputDataLines", "ma~") YIELD node, score RETURN node, score```
END_OF_EXAMPLE_5


- Input question: {question}
- Cypher query:"""


TEXT_RAG_PROMPT = PromptTemplate(
    input_variables=["question"], template=TEXT_RAG_TEMPLATE
)

###

COMBINED_RAG_TEMPLATE = """You have access to two different sources of information retrieved from the original data:

- Source 1: A retrieval result produced from a text search on the original data.
- Source 2: A retrieval result produced from querying a knowledge graph created from the original data.

Your task is to read the context from both sources and select the most relevant information to answer the question provided. Follow these steps:

1. Read and analyze the context from both sources, noting the differences in how the information was retrieved (text search vs. knowledge graph query).
2. Compare the information to identify key points, similarities, and differences between the sources.
3. Select the most relevant information from both sources that address the question, prioritizing completeness.
4. Answer the question clearly and concisely, integrating the selected information from both sources.

If the sources provide conflicting information, use your best judgment to determine which points are most credible or relevant, and mention any discrepancies in your response if necessary.

Reply only with the correct answer --no explanations about the source context you used.

Source 1 context:
{text_rag_context}

Source 2 context:
{graph_rag_context}

Question: {question}
"""

COMBINED_RAG_PROMPT = ChatPromptTemplate.from_template(COMBINED_RAG_TEMPLATE)

###

HYBRID_RAG_TEMPLATE = """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query that retrieves the relevant answer based on the graph schema. Optionally, you may use the text search functionality provided in the documentation to enhance the query. 

### Steps:
1. Understand the question and decide if a fulltext search is relevant to include in the query. 
2. If a fulltext search is needed, refer to the provided documentation to construct the search expression.
3. Construct a syntactically correct Cypher query using the graph schema provided, and the fulltext search on the `inputDataLines` index if applicable.
4. Ensure that characters `:`, `/`, and `-` are always escaped with `\` when constructing the query.


### Documentation for Fulltext Search:
- Specific attribute (complete match):
EXAMPLE_1:
- Input question: Search for movies that have the word “matrix” in their title.
- Cypher query: 
```CALL db.index.fulltext.queryNodes("inputDataLines", "matrix") YIELD node, score RETURN node, score```
END_OF_EXAMPLE_1

- Logical operator:
There are two logical operators available, “OR” and “AND”.
EXAMPLE_2:
- Input question: Search for movies that have the words "war" OR "matrix: revolution" in their title.
- Cypher query:
```CALL db.index.fulltext.queryNodes("inputDataLines", "'the' OR 'matrix\: revolution'") YIELD node, score RETURN node, score```
END_OF_EXAMPLE_2

- Single-character wildcard:
The single-character wildcard operator `?` looks for terms that match that with the single character replaced.
EXAMPLE_3:
- Input question: Can you show me movies whose title begins with 'ma'?
- Cypher query:
```CALL db.index.fulltext.queryNodes("inputDataLines", "ma?") YIELD node, score RETURN node, score```
END_OF_EXAMPLE_3

- Multi-character wildcard:
The multi-character wildcard `*` operator looks for zero or more characters. It can be used before, after, or in the middle of an expression.
EXAMPLE_4:
- Input question: What are movies with title containing 'sun'?
- Cypher query:
```CALL db.index.fulltext.queryNodes("inputDataLines", "*sun*") YIELD node, score RETURN node, score```
END_OF_EXAMPLE_4

- Fuzzy search:
Fuzzy search `~` works by using mathematical formulae that calculate the similarity between two words.
EXAMPLE_5:
- Input question: Can you find movies whose titles are similar to "matrix"?
- Cypher query:
```CALL db.index.fulltext.queryNodes("inputDataLines", "ma~") YIELD node, score RETURN node, score```
END_OF_EXAMPLE_5


### Input:
- Schema Information: {schema}
- User Question: {question}

### Output:
Respond with the Cypher query only -- no other text or explanations.

Cypher query:"""


HYBRID_RAG_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=HYBRID_RAG_TEMPLATE
)