import os

ROOT = os.getcwd()
FOLDER_IMAGES = os.path.join(ROOT, 'pdf_data/images')
FOLDER_PDF = os.path.join(ROOT,'pdf_data/pdf')
FOLDER_META = os.path.join(ROOT,'pdf_data/meta')
PATH_CHUNKS = os.path.join(FOLDER_META,'chunks.parquet')
PATH_METADATA = os.path.join(FOLDER_META,'meta_docs.json')
PATH_VECTOR = os.path.join(ROOT,'vector')
PATH_LOGO = os.path.join(ROOT,'static')
os.makedirs(FOLDER_IMAGES, exist_ok=True)
os.makedirs(FOLDER_PDF, exist_ok=True)
os.makedirs(FOLDER_META, exist_ok=True)