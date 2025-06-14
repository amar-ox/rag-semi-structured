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


from .rag_prompt_templates import (
    GRAPH_RAG_PROMPT,
    TEXT_RAG_PROMPT,
    COMBINED_RAG_PROMPT,
    HYBRID_RAG_PROMPT
)
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class FastRAG:
    def __init__(self, url, username, password, llm):
        self._graph = Neo4jGraph(
            url=url, 
            username=username, 
            password=password)
        self._schema = self._graph.schema
        self._llm = llm
        
    @property
    def schema(self):
        return self._schema
    
    def close(self):
        self._graph._driver.close()

    def build_graph_chain(self, top_k=10, return_direct=False, verbose=False):
        graph_chain = GraphCypherQAChain.from_llm(
            self._llm, 
            graph=self._graph,
            exclude_types=['InputLine','HAS_INPUT_LINE'],
            validate_cypher=True,
            verbose=verbose,
            cypher_prompt=GRAPH_RAG_PROMPT,
            return_direct=return_direct,
            top_k=top_k
        )
        return graph_chain

    def build_text_chain(self, top_k=30, return_direct=False, verbose=False):
        text_chain = GraphCypherQAChain.from_llm(
            self._llm,
            graph=self._graph,
            #  exclude_types=[],
            validate_cypher=True,
            verbose=verbose,
            cypher_prompt=TEXT_RAG_PROMPT,
            return_direct=return_direct,
            top_k=top_k
        )
        return text_chain
        
    def build_hybrid_chain(self, top_k=10, return_direct=False, verbose=False):
        hybrid_chain = GraphCypherQAChain.from_llm(
            self._llm, 
            graph=self._graph,
            # exclude_types=[],
            validate_cypher=True,
            verbose=verbose,
            cypher_prompt=HYBRID_RAG_PROMPT,
            return_direct=return_direct,
            top_k=top_k
        )
        return hybrid_chain


    def build_combined_chain(self, verbose=False):
        # create text_chain with direct results:
        direct_t_c = self.build_text_chain(return_direct=True, verbose=verbose)

        # create graph_chain with direct results:
        direct_g_c = self.build_graph_chain(return_direct=True, verbose=verbose)

        combined_chain = RunnableMap(
            {
                "query": RunnablePassthrough(),
                "result": (
                    {"text_rag_context": direct_t_c, "graph_rag_context": direct_g_c, "question": RunnablePassthrough()}
                    | COMBINED_RAG_PROMPT
                    | self._llm
                    | StrOutputParser()
                ),
            }
        )
        return combined_chain
    
    @classmethod
    def generate_questions(self, llm, chunks):
        qa_data = '\n'.join(chunks)
        prompt = f"""You are a data scientis. Your task is to generate an evaluation Q&A dataset from a sample of data. 
Use the given data to generate a set of general questions and answers that can be used on the original complete data of this sample.

IMPORTANT: Make sure the questions are also relevant for the original complete data from which the sample data is taken.  

Data:
```
{qa_data}
```

Each question and answer must be in a single line. Start each question with `Q: ` and each answer with `A: `"""
        print(prompt)
        output = llm.model.invoke(prompt)
        generated_questions = output.content
        return generated_questions
