import click
import torch as th 

from os import path 
from glob import glob 
from pydantic import BaseSettings

from map2gpt.runner import GPTRunner
from map2gpt.deploy import deploy_on_telegram
from qdrant_client import QdrantClient

class EnvVars(BaseSettings):
    OPENAI_API_KEY: str
    TRANSFORMERS_CACHE: str
    QDRANT_PERSISTENT_FOLDER: str

@click.command()
@click.option('--path2directory', type=click.Path(exists=True, dir_okay=True), required=True)
def main(path2directory: str):
    envvars = EnvVars()  # OPENAI_API_KEY, TRANSFORMERS_CACHE, QDRANT_PERSISTENT_FOLDER
    
    text_filepaths = sorted(glob(path.join(path2directory, '*.txt')))
    text_contents = []
    for text_filepath in text_filepaths[:30]:
        with open(text_filepath, 'r') as file_pointer:
            text_contents.append(file_pointer.read())
    
    qdrant = QdrantClient(path=envvars.QDRANT_PERSISTENT_FOLDER)
    device = th.device('cpu' if not th.cuda.is_available() else 'cuda')

    runner = GPTRunner(
        qdrant=qdrant,
        device=device,
        openai_api_key=envvars.OPENAI_API_KEY,
        tokenizer='gpt-3.5-turbo',
        transformers_cache=envvars.TRANSFORMERS_CACHE,
        transformer_model_name='Sahajtomar/french_semantic',
    )

    knowledge_base = runner.build_index_from_texts(
        texts=text_contents,
        chunk_size=256,
        batch_size=128,
        name='OUEST-FRANCE',
    )

    runner.create_qdrant_index(knowledge_base=knowledge_base)

    deploy_on_telegram(
        runner=runner,
        name='OUEST-FRANCE',
        description="""
            Un ChatBOT qui répond à vos questions sur l'actualité de l'Ouest de la France.
            Il est basé sur les articles de l'Ouest-France.
        """,
        telegram_token='6098806720:AAHFhm9w4HnVQTBC_PgoZo6Wzk-dN_jqDVk',
        top_k=11,
        source_k=3
    )



if __name__ == '__main__':
    main()