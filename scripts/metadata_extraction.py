from typing import Union, Sequence, TypeVar, Type

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Node, BaseNode, TransformComponent
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core.extractors import PydanticProgramExtractor, BaseExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.types import Model
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class RAGMetadata(BaseModel):
    """Metadata for text chunk"""

    title: Union[str, None] = Field(default=None, description="The title of the document")
    power: Union[int, None] = Field(
        default=0, description="If technical details for a bulb are present the power of the bulb else 0"
    )
    lifespan: Union[int, None] = Field(
        default=0, description="If technical details for a bulb are present  the lifespan of the bulb else 0"
    )


def build_extractor(metadata_model: Type[Model] = RAGMetadata) -> BaseExtractor:
    """Builds the transformation component for metadata extraction

    Args:
        metadata_model (Type(BaseModel)): The model to use for metadata extraction

    Returns:
        transform_component (
    """
    openai_program = OpenAIPydanticProgram.from_defaults(
        output_cls=metadata_model,
        prompt_template_str="{input}",
    )

    return PydanticProgramExtractor(
        program=openai_program, input_key="input", show_progress=True
    )


def get_transformations(
        metadata_model: Type[Model] = RAGMetadata,
        node_parser: SentenceSplitter = SentenceSplitter(chunk_size=1024)
) -> list[TransformComponent]:
    """Get the transformations for metadata extraction

    Args:
        metadata_model (Type(BaseModel)): The model to use for metadata extraction
        node_parser (SentenceSplitter): The node parser to use for splitting documents into nodes

    Returns:
        transformations (list[TransformComponent]): The transformations for metadata extraction
    """
    extractor = build_extractor(metadata_model=metadata_model)
    return [node_parser, extractor]


def extract_metadata(
        nodes: list[BaseNode] = None,
        docs: list[Document] = None,
        metadata_model: Type[Model] = RAGMetadata) -> Sequence[BaseNode]:
    """Extracts metadata from the document or nodes.

    Args:
        nodes (list[Node]): The nodes to extract metadata from
        docs (list[Document]): The documents to extract metadata from
        metadata_model (Type(BaseModel)): The model to use for metadata extraction

    Returns:
        new_nodes (list[Node]): The nodes with metadata extracted
    """
    extractor = build_extractor(metadata_model=metadata_model)

    if nodes:
        return extractor(nodes)

    assert docs is not None, "Documents must be provided if nodes are not provided"

    node_parser = SentenceSplitter(chunk_size=1024)
    ingestion_pipeline = IngestionPipeline(transformations=[node_parser, extractor])
    return ingestion_pipeline.run(documents=docs)
