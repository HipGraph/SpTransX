import sqlite3
import sys
from tqdm import tqdm 
import gc
import os
 

col_separator = ' '
# Function to create the SQLite3 database and tables
def create_database(db_name):
    if os.path.exists(db_name):
        print(f"Warning! Database {db_name} already exists. Overwriting...")
        os.remove(db_name)
        
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Create table for triplets
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS triplets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id INTEGER,
            predicate_id INTEGER,
            object_id INTEGER
        )
    ''')
    
    # Create table for unique entities
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entities (
            entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity TEXT UNIQUE
        )
    ''')
    
    # Create table for unique relations
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS relations (
            relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            relation TEXT UNIQUE
        )
    ''')
    
    conn.commit()
    conn.close()



def get_entity_id(cursor, entity):
    # First, check if the relation exists
    cursor.execute('''
        SELECT entity_id FROM entities
        WHERE entity = ?
    ''', (entity,))
    result = cursor.fetchone()
    
    if result:
        # If relation exists, return the relation_id
        return result[0] - 1
    else:
        # If relation doesn't exist, insert it and return the new relation_id
        cursor.execute('''
            INSERT INTO entities (entity)
            VALUES (?)
        ''', (entity,))
        return cursor.lastrowid - 1

def get_relation_id(cursor, relation):
    # First, check if the relation exists
    cursor.execute('''
        SELECT relation_id FROM relations
        WHERE relation = ?
    ''', (relation,))
    result = cursor.fetchone()
    
    if result:
        # If relation exists, return the relation_id
        return result[0] - 1
    else:
        # If relation doesn't exist, insert it and return the new relation_id
        cursor.execute('''
            INSERT INTO relations (relation)
            VALUES (?)
        ''', (relation,))
        return cursor.lastrowid - 1

def process_line(x):
    global col_separator
    items = x.split(col_separator, 2)
    # entities.add(items[0].strip())
    # entities.add(items[2][:-3].strip())
    # rel.add(items[1].strip())
    return items[0].strip(), items[1].strip(), items[2].strip()


# Function to insert triplets into the database in batches
def insert_triplets_batch(db_name, triplet_file, num_lines, batch_size=1000):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    with open(triplet_file, 'r') as file:
        batch = []
        with tqdm(total=num_lines) as pbar:
            for _ in range(num_lines):
                line = file.readline()
                if len(batch) >= batch_size:
                    # Insert current batch
                    insert_batch(cursor, batch)
                    del batch
                    gc.collect()
                    batch = []
                    # break

                subject, predicate, obj = process_line(line)

                subject_id = get_entity_id(cursor, subject)
                predicate_id = get_relation_id(cursor, predicate)
                object_id = get_entity_id(cursor, obj)

                batch.append((subject_id, predicate_id, object_id))
                pbar.update(1)
            # Insert any remaining triplets
            if batch:
                insert_batch(cursor, batch)
    
    conn.commit()
    conn.close()

# Function to insert a batch of triplets into the database
def insert_batch(cursor, batch):
    cursor.executemany('''
        INSERT INTO triplets (subject_id, predicate_id, object_id)
        VALUES (?, ?, ?)
    ''', batch)

    
def count_lines_in_file(file_path):
    with open(file_path, 'r') as file:
        line_count = sum(1 for line in file)
    return line_count

# Example usage
def convert_nt_to_db(filename, batch_size, num_lines=None, db_name=None, sep=','):
    global col_separator
    col_separator = sep
    num_lines_given = num_lines or count_lines_in_file(filename)
    db_name_given = db_name or filename.split('.')[0] + '_' + str(num_lines_given) + '.db'
    create_database(db_name_given)
    print(f'db created: {db_name_given}')
    insert_triplets_batch(db_name_given, filename, num_lines_given, batch_size=batch_size)
    print("Triplets and unique entities/relations have been stored in the database in batches.")

# num_lines = int(sys.argv[1])
# batch_size = int(sys.argv[2])

# convert_nt_to_db('train.txt', 50000, num_lines=None, sep='\t', db_name='fb15k.db')