# MAP2GPT

## description 

This project is a versatile and powerful search tool that leverages state-of-the-art natural language processing models to provide relevant and contextually rich results. The primary goal of this project is to build a semantic search engine for textual content from various sources such as PDF files and Wikipedia pages.

The project utilizes the GPT-3.5-turbo model for generating responses and French Semantic model to create embeddings of textual data. Users can build an index of embeddings from a PDF file or a Wikipedia page, explore the index interactively, and deploy the search functionality on Telegram. The search results are presented as the top k relevant chunks of information, which are then used as context to generate an informative response from the GPT-3.5-turbo model.

The project is implemented in Python, and it employs several open-source libraries such as Click, OpenAI, Wikipedia, PyTorch, Tiktoken, and Rich. The code is organized into modular functions and classes, making it easy to understand, maintain, and extend. The main script provides a command-line interface for users to interact with the project's functionalities.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
   - [Build Index from PDF](#build-index-from-pdf)
   - [Build Index from Wikipedia](#build-index-from-wikipedia)
   - [Explore Index](#explore-index)

## Installation

To install the necessary dependencies, run the following command:

```bash
python -m venv env 
source env/bin/activate
pip install --upgrade pip 
pip install map2gpt 
```

## Supported Transformer Models

This project supports a variety of transformer models, including models from the Hugging Face Model Hub and sentence-transformers. Below are some examples:
    - Hugging Face Model: 'Sahajtomar/french_semantic'
    - Sentence-Transformers Model: 'paraphrase-MiniLM-L6-v2', 'all-mpnet-base-v2', etc...

Please ensure that the model you choose is compatible with the project requirements and adjust the `--transformer_model_name` option accordingly.

# CLI usage 

## set env vars 
```bash
    export OPENAI_API_KEY=sk- TRANSFORMERS_CACHE=/path/to/cache QDRANT_PERSISTENT_FOLDER=/path/to_persistent
```

## Build Index from PDF files
To build an index from a PDF file, run the following command:

```bash
python -m map2gpt.main --transformer_model_name 'Sahajtomar/french_semantic' build-index-from-pdf-files
    --path2pdf_files /path/to/file-000.pdf \
    --path2pdf_files /path/to/file-001.pdf \
    --name qdrant_collection_name \
    --chunk_size 256 \
    --batch_size 128
```

## Build Index from Wikipedia pages
To build an index from a Wikipedia page, run the following command:

```bash
python -m map2gpt.main --transformer_model_name 'Sahajtomar/french_semantic' build-index-from-wikipedia-pages
    --urls https://...wikipedia \
    --urls https://...wikipedia \
    --name qdrant_collection_name \
    --chunk_size 256 \
    --batch_size 128
```

## Build Index from Youtube links 
To build an index from a Wikipedia page, run the following command:

```bash
python -m map2gpt.main --transformer_model_name 'Sahajtomar/french_semantic' build-index-from-youtube-links
    --urls https://...youtube \
    --urls https://...youtube \
    --name qdrant_collection_name \
    --chunk_size 256 \
    --batch_size 128
```

## Build Index from texts
To build an index from a Wikipedia page, run the following command:


```bash
python -m map2gpt.main --transformer_model_name 'Sahajtomar/french_semantic' build-index-from-wikipedia-pages
    --path2directory /path/to/corpus_text_files
    --name qdrant_collection_name \
    --chunk_size 256 \
    --batch_size 128
```

# Explore Index
To explore the index, run the following command:


## query the index

```bash
python -m map2gpt.main --transformer_model_name 'Sahajtomar/french_semantic' query-index
    --query "...." \
    --name qdrant_collection_name \ 
    --top_k 7
    --source_k 3
    --description "service description"
```

## deploy on telegram 

```bash
python -m map2gpt.main --transformer_model_name 'Sahajtomar/french_semantic' deploy-on-telegram
    --telegram_token XXXXXXXXX...XXXXXXXXXXX \
    --name qdrant_collection_name \ 
    --top_k 7
    --source_k 3
    --description "service description"
```

# Module usage  
```python
    # create qdrant client 
    qdrant = QdrantClient(':memory:') # use path for persistence QdrantClient(path=path2persistent_dir)
    
    # initialize runner
    runner = GPTRunner(
        device='cuda:0',  # cpu
        qdrant=qdrant,
        tokenizer='gpt-3.5-turbo',
        openai_api_key='sk-XXXXXXXXXXXXXXXXXXXXX',
        transformers_cache='/path/to/transformers_cache',
        transformer_model_name='Sahajtomar/french_semantic'  # use all-mpnet-case-v2 for english
    )

    # build index from wikipedia pages
    knowledge_base = runner.build_index_from_pdf_files(
        path2pdf_files=[
            'https://www.youtube.com/watch?v=tH-i_FeagJc',
            'https://www.youtube.com/watch?v=tH-i_FeagJc',
        ],
        chunk_size=256,
        batch_size=128,
        name='collection_name',
    )
    
    # create qdrant index
    runner.create_qdrant_index(knowledge_base=knowledge_base)

    # deploy on telegram
    deploy_on_telegram(
        telegram_token='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', 
        runner=runner, 
        name='collection_name', 
        description="service name description", 
        top_k=10, 
        source_k=3
    )
```