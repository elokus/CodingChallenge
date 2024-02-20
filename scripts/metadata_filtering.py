from llama_index.core import Response
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.base import BaseIndex
from llama_index.core.vector_stores import MetadataFilter
from llama_index.legacy.vector_stores import MetadataFilters

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def filter_metadata_by_operator(metadata, filter_: MetadataFilter) -> bool:
    if filter_.operator == "<=":
        if metadata[filter_.key] <= filter_.value:
            return True
    elif filter_.operator == ">=":
        if metadata[filter_.key] >= filter_.value:
            return True
    elif filter_.operator == "<":
        if metadata[filter_.key] < filter_.value:
            return True
    elif filter_.operator == ">":
        if metadata[filter_.key] > filter_.value:
            return True
    elif filter_.operator == "==":
        if metadata[filter_.key] == filter_.value:
            return True
    elif filter_.operator == "!=":
        if metadata[filter_.key] != filter_.value:
            return True
    return False


def filters_metadata_entry(metadata: dict, filters: MetadataFilters) -> bool:
    if all([filter_metadata_by_operator(metadata, filter_) for filter_ in filters.filters]):
        return True
    return False


def filter_metadata_dict(metadata_dict: dict, filters: MetadataFilters) -> dict:
    metadata_filtered = {}
    for node_id, metadata in metadata_dict.items():
        if filters_metadata_entry(metadata, filters):
            metadata_filtered[node_id] = metadata

    return metadata_filtered


def get_filtered_node_ids(index: BaseIndex, filters: MetadataFilters) -> list:
    metadata_dict = index.vector_store._data.metadata_dict
    metadata_filtered = filter_metadata_dict(metadata_dict, filters)
    return list(metadata_filtered.keys())


def query_engine_from_filters(index: BaseIndex, filters: MetadataFilters, top_k: int = 10) -> BaseQueryEngine:
    filtered_node_ids = get_filtered_node_ids(index, filters)
    engine = index.as_query_engine(similarity_top_k=top_k)
    engine.retriever._node_ids = filtered_node_ids
    return engine


def filters_from_query(query_str: str) -> MetadataFilters:

    metadata_keys = {"power": "int", "lifespan": "int", "color": "str"}

    prompt = PromptTemplate(
        template=("If the given query mentions instruction for a suitable metadata filter "
                  "I want you to prepare metadata filters for the following metadata keys: {metadata_keys}\n"
                  "The format of the metadata filters should be as follows: \n"
                  "Return a valid json with {schema} \n"
                  "valid operators are: '==', '>', '<', '!=', '>=', '<=', 'in', 'nin', 'text_match'\n"
                  "If the query does not mention any instruction for metadata filters, return an empty dictionary\n"
                  "Please return a valid json! Your output should be parsable by the json parser"
                  "The query is: {query}"),
        metadata_keys=metadata_keys,
        input_variables=["query", "metadata_keys"],
        partial_variables={"schema": "\"{'filters': [{key: str, value: Union[str, int], operator: str}, ...]}\""}
    )

    parser = PydanticOutputParser(pydantic_object=MetadataFilters)

    model = ChatOpenAI(temperature=0)
    chain = prompt | model | parser

    result = chain.invoke(dict(query=query_str, metadata_keys=metadata_keys))

    return result


def query_with_metadata_filters(index: BaseIndex, query: str, top_k: int = 10, retry: int = 3) -> Response:
    i = 0
    while i <= retry:
        if i == retry:
            raise Exception("Failed to parse metadata filters")
        try:
            filters = filters_from_query(query)
            break
        except Exception as e:
            i += 1

    engine = query_engine_from_filters(index, filters, top_k=top_k)
    return engine.query(query)
