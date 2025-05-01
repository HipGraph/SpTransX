def ttl_to_csv(ttl_filepath, csv_filepath):
    """
    Converts a knowledge graph in TTL format to a CSV file with 'from', 'rel', and 'to' columns.

    Args:
        ttl_filepath (str): The path to the input TTL file.
        csv_filepath (str): The path to the output CSV file.  Will overwrite if it exists.
    """
    import rdflib
    from urllib.parse import unquote

    try:
        g = rdflib.Graph()
        g.parse(ttl_filepath, format="ttl")

        with open(csv_filepath, 'w', encoding='utf-8') as f:
            f.write("from\trel\tto\n")
            for s, p, o in g:
                # Decode URI-encoded strings
                from_node = unquote(str(s))
                rel = unquote(str(p))
                to_node = unquote(str(o))
                f.write(f"{from_node}\t{rel}\t{to_node}\n")
        print(f"Successfully converted TTL to CSV: {csv_filepath}")

    except Exception as e:
        print(f"Error converting TTL to CSV: {e}")

def rdf_to_csv(rdf_filepath, csv_filepath):
    """
    Converts a knowledge graph in RDF/XML format to a CSV file with 'from', 'rel', and 'to' columns.

    Args:
        rdf_filepath (str): The path to the input RDF/XML file.
        csv_filepath (str): The path to the output CSV file. Will overwrite if it exists.
    """
    import rdflib
    from urllib.parse import unquote
    try:
        g = rdflib.Graph()
        g.parse(rdf_filepath, format="xml") # auto detects also json-ld

        with open(csv_filepath, 'w', encoding='utf-8') as f:
            f.write("from\trel\tto\n")
            for s, p, o in g:
                # Decode URI-encoded strings
                from_node = unquote(str(s))
                rel = unquote(str(p))
                to_node = unquote(str(o))
                f.write(f"{from_node}\t{rel}\t{to_node}\n")
        print(f"Successfully converted RDF/XML to CSV: {csv_filepath}")
    except Exception as e:
        print(f"Error converting RDF/XML to CSV: {e}")



def neo4j_to_csv(uri, username, password, csv_filepath):
    """
    Converts a knowledge graph from a Neo4j database to a CSV file with 'from', 'rel', and 'to' columns.

    Args:
        uri (str): The URI of the Neo4j database (e.g., "bolt://localhost:7687").
        username (str): The username for connecting to the Neo4j database.
        password (str): The password for connecting to the Neo4j database.
        csv_filepath (str): The path to the output CSV file.
    """
    from neo4j import GraphDatabase
    from urllib.parse import unquote

    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session() as session:
            query = """
                MATCH (n)-[r]->(m)
                RETURN n.id, type(r), m.id
                """  # Changed to n.id and m.id

            result = session.run(query)

            with open(csv_filepath, 'w', encoding='utf-8') as f:
                f.write("from\trel\tto\n")
                for record in result:
                    # Ensure record has the expected structure and handle None values
                    from_node = unquote(str(record[0])) if record[0] is not None else "None"
                    rel = unquote(str(record[1])) if record[1] is not None else "None"
                    to_node = unquote(str(record[2])) if record[2] is not None else "None"
                    f.write(f"{from_node}\t{rel}\t{to_node}\n")
        driver.close()
        print(f"Successfully converted Neo4j to CSV: {csv_filepath}")

    except Exception as e:
        print(f"Error converting Neo4j to CSV: {e}")
