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


# fastrag/graph_creator.py

from neo4j import GraphDatabase


class KnowledgeGraphCreator:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._initialize_database()
     
    def clean_database(self):
        with self.driver.session() as session:
            session.execute_write(self._delete_all_nodes)

    def create_graph(self, data):
        with self.driver.session() as session:
            for file_name, file_content in data.items():
                session.write_transaction(self._create_file_node, file_name, file_content)
                
    def close(self):
        self.driver.close()

    def _initialize_database(self):
        with self.driver.session() as session:
            session.run("DROP INDEX inputDataLines IF EXISTS")
            session.run("CREATE FULLTEXT INDEX inputDataLines FOR (n:InputLine) ON EACH [n.content]")
            session.run("MATCH (n) DETACH DELETE n")
                
    def _delete_all_nodes(self, tx):
        delete_query = """MATCH (n) DETACH DELETE n"""
        tx.run(delete_query)

    def _create_file_node(self, tx, file_name, file_content):
        tx.run("MERGE (file:File {name: $file_name}) RETURN file", file_name=file_name)
        for key, objects in file_content.items():
            for obj in objects:
                self._create_node_recursive(tx, obj, key, file_name, True)

    def _create_node_recursive(self, tx, obj, label, parent_id, first):
        # get all props that are not a nested object
        props = {k: v for k, v in obj.items() if isinstance(v, (str, int, float, bool, list)) and k != 'input_data'}
        #print(f"=props: {props}")

        # get all props that are a list of objects
        list_of_dict_props = {k: v for k, v in obj.items() if isinstance(v, list) and v and all(isinstance(i, (dict)) for i in v)}
        #print(f"=loo props: {list_of_dict_props}")

        # keep only simple of list of simple props
        simple_props = {k: v for k, v in props.items() if k not in list_of_dict_props}
        #print(f"=simple props: {simple_props}")

        # Create the node and process input_data at the same time
        node_query = f"""
        CREATE (n:{label} {{ {', '.join(f'{k}: ${k}' for k in simple_props)} }})
        RETURN id(n) as node_id"""
        #print(node_query)

        result = tx.run(node_query, **props)
        node_id = result.single()['node_id']

        if first:   # 1-st level nodes
            relationship_query = f"""
            MATCH (parent:File {{name: $parent_id}})
            MATCH (child) WHERE id(child) = $node_id
            MERGE (parent)-[:CONTAINS]->(child)
            """
            tx.run(relationship_query, parent_id=parent_id, node_id=node_id)
        else:          # other nodes
            relationship_query = f"""
            MATCH (parent) WHERE id(parent) = $parent_id
            MATCH (child) WHERE id(child) = $node_id
            MERGE (parent)-[:CONTAINS]->(child)
            """
            tx.run(relationship_query, parent_id=parent_id, node_id=node_id)

        # Handle and process input_data if present
        input_data = obj.get('input_data')
        if input_data:
            input_lines = set(input_data.split('\n'))
            for line in input_lines:
                if line.strip():
                    self._create_input_line_node(tx, line.strip(), node_id)

        # Handle nested objects and lists of objects
        for key, value in obj.items():
            if isinstance(value, dict):
                #print(f"->insert dict prop: {key}")
                self._create_node_recursive(tx, value, key, node_id, False)
            elif isinstance(value, list) and any(isinstance(i, dict) for i in value):
                #print(f"->insert list of dict prop: {key}")
                for item in value:
                    if isinstance(item, dict):
                        self._create_node_recursive(tx, item, key, node_id, False)

    def _create_input_line_node(self, tx, line, parent_node_id):
        result = tx.run("CREATE (n:InputLine {content: $line}) RETURN id(n) as input_line_id", line=line)
        input_line_id = result.single()['input_line_id']
        relationship_query = """
            MATCH (parent) WHERE id(parent) = $parent_node_id
            MATCH (child) WHERE id(child) = $input_line_id
            MERGE (parent)-[:HAS_INPUT_LINE]->(child)
        """
        tx.run(relationship_query, parent_node_id=parent_node_id, input_line_id=input_line_id)
        

        

