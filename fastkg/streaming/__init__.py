import os

supported_modes = ['sqlite3']
# mode = 'sqlite3'
mode = os.getenv('FASTKG_STREAM_DB_ENGINE', 'sqlite3')

if mode == 'sqlite3':
    print('FastKG streaming db engine: SQLite3')
    from .sqlt3.manager import convert_nt_to_db
    from .sqlt3.utils import StreamingSparseKGDataset, map_entity_and_rel
else:
    print(f"Error! Found unsupported streaming DB mode: '{mode}' in FASTKG_STREAM_DB_ENGINE environment variable.\nOnly the followings modes are supported for streaming at this moment:\n\t {','.join(supported_modes)}")
    print('Run the following command to set sqlite3 as db engine:\n\t export FASTKG_STREAM_DB_ENGINE="sqlite3"')
    exit()

__all__ = ['convert_nt_to_db', 'StreamingSparseKGDataset', 'map_entity_and_rel']