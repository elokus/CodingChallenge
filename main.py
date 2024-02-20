from typing import Tuple

from llama_index.core import StorageContext, Response
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import TransformComponent
from llama_index.core.extractors import (
    TitleExtractor,
)
from llama_index.core.node_parser import TokenTextSplitter

from scripts.metadata_extraction import build_extractor
from scripts.core import initialize_index, initialize_storage, load_document, simple_query
from scripts.metadata_filtering import query_with_metadata_filters
import os


def prepare_index(
        file_paths: list[str],
        load_dir: str = None,
        transformations: list[TransformComponent] = None
) -> Tuple[BaseIndex, StorageContext]:

    # prepare transformations
    if transformations is None:
        transformations = [
            TokenTextSplitter(
                separator=" ", chunk_size=512, chunk_overlap=128),
            TitleExtractor(nodes=1),
            build_extractor()
        ]

    # load or initialize index
    if load_dir:
        storage = initialize_storage(load_dir=load_dir)
        index = initialize_index(storage, load_from_storage=True)
    else:
        docs = []
        for file_path in file_paths:
            docs.extend(load_document(file_path))

        storage = initialize_storage()
        index = initialize_index(storage, docs, transformations=transformations)

    return index, storage


def query(query: str, index: BaseIndex, top_k=2) -> Response:
    response = simple_query(index, query, top_k)
    stdout_response(response, top_k=top_k)
    return response


def query_with_metadata(query: str, index: BaseIndex, top_k=2) -> Response:
    response = query_with_metadata_filters(index, query, top_k=top_k)
    stdout_response(response, top_k=top_k)
    return response


def stdout_response(response: Response, top_k=2):
    print("=== Antwort ===")
    print(response.response)
    print(f"\n=== Quellen (Top {top_k}) ===")

    for node in response.source_nodes:
        print("Metadata:")
        print(node.metadata)
        print("Quellcontent:")
        print(node.text[:1000])
        print("-----------------------------------\n")





if __name__ == "__main__":


    persist_index_dir = "challenge_index"

    # create or load index
    if not os.path.exists(persist_index_dir):

        document_dir = "data/coding_challenge"
        file_paths = [os.path.join(document_dir, file) for file in os.listdir(document_dir)]

        index, storage = prepare_index(file_paths)
        storage.persist(persist_index_dir)

    else:
        index, storage = prepare_index([], load_dir=persist_index_dir)

    # Teil 1:
    query_str = "Wie ist die Lebensdauer der XBO 2000 W/HTP XL OFR-Lampe?"
    result = query(query_str, index)

    query_str = "Gebe mir alle Leuchtmittel mit mindestens 1500W und einer Lebensdauer von mehr als 3000 Stunden"
    # Teil 2:
    result = query_with_metadata(query_str, index, top_k=5)
