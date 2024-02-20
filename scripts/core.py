from typing import Union

from llama_index.core.data_structs import Node
from llama_index.core.extractors import BaseExtractor
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import TransformComponent
from llama_index.core.vector_stores.types import VectorStore
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage, Response, \
    SimpleDirectoryReader
from dotenv import load_dotenv
import os

load_dotenv()


def initialize_storage(vector_store: VectorStore = None, load_dir: str = None) -> StorageContext:
    """Initializes the storage context for the index

    If a load directory is provided, the vector store is loaded from the directory.
    Otherwise, a new storage context is created with the provided vector store.
    If a vector store is not provided, a simple vector store is created.

    Args:
        vector_store (VectorStore, optional): The vector store to use. Defaults to None.
        load_dir (str, optional): The path to load the vector store from. Defaults to None.

    Returns:
        StorageContext: The storage context for the index
    """
    if load_dir:
        return StorageContext.from_defaults(persist_dir = load_dir)

    if vector_store is None:
        vector_store = SimpleVectorStore()

    return StorageContext.from_defaults(vector_store=vector_store)


def initialize_index(
        storage_context: StorageContext,
        docs: list[Document] = None,
        transformations: list[TransformComponent] = None,
        load_from_storage: bool = False
) -> BaseIndex:
    """Initializes the index with the provided storage context and documents

    If the load_from_storage flag is set to True, the index is loaded from the storage context.

    Args:
        storage_context (StorageContext): The storage context for the index
        docs (list[Document]): The documents to add to the index
        load_from_storage (bool): Whether to load the index from storage

    Returns:
        VectorStoreIndex: The initialized index
    """

    if load_from_storage:
        return load_index_from_storage(storage_context)

    assert docs is not None, "Documents must be provided if the index is not loaded from storage"
    assert os.getenv("OPENAI_API_KEY") is not None, "OpenAI API key must be set in the environment"

    return VectorStoreIndex.from_documents(docs, storage_context, transformations=transformations)


def load_document(file_path: str) -> list[Document]:
    reader = SimpleDirectoryReader(input_files=[file_path])
    docs = reader.load_data(file_path)
    return docs

def load_documents(document_dir: str) -> list[Document]:
    return SimpleDirectoryReader(input_dir=document_dir).load_data()


def add_documents(index: BaseIndex, docs: Union[list[Document], Document]) -> BaseIndex:
    docs = [docs] if isinstance(docs, Document) else docs
    for doc in docs:
        index.insert(doc)
    return index


def add_nodes(index: BaseIndex, nodes: Union[list[Node], Node]) -> BaseIndex:
    nodes = [nodes] if isinstance(nodes, Node) else nodes
    index.insert_nodes(nodes)
    return index


def simple_query(index: BaseIndex, query: str, top_k=2) -> Response:
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    response = query_engine.query(query)
    return response

