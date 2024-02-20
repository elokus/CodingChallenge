import pytest
from scripts.core import initialize_index, initialize_storage, load_document


def test_initialize_index():
    test_file = "../data/TestFile.pdf"
    docs = load_document(test_file)
    storage = initialize_storage()
    index = initialize_index(storage, docs)

    assert len(index._docstore.docs) == 37


def test_load_index():
    storage = initialize_storage(load_dir="../test_index")
    index = initialize_index(storage, load_from_storage=True)

    assert len(index._docstore.docs) == 37
