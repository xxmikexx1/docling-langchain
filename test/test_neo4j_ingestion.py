from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from langchain_docling.neo4j import ingest_pdfs_to_neo4j


class DummyLoader:
    def __init__(self, file_path, convert_kwargs=None, chunker=None):
        from docling.chunking import HybridChunker

        assert isinstance(chunker, HybridChunker)
        self._docs = [Document(page_content="chunk", metadata={})]

    def lazy_load(self):
        yield from self._docs


def test_ingest_pdfs_to_neo4j(monkeypatch: pytest.MonkeyPatch) -> None:
    embedding = MagicMock()
    neo4j_vs = MagicMock()
    monkeypatch.setattr("langchain_docling.neo4j.DoclingLoader", DummyLoader)
    monkeypatch.setattr(
        "langchain_docling.neo4j.Neo4jVector.from_documents",
        MagicMock(return_value=neo4j_vs),
    )

    res = ingest_pdfs_to_neo4j(
        file_paths=["file.pdf"],
        embedding=embedding,
        url="bolt://localhost:7687",
        username="neo4j",
        password="pw",
    )

    from langchain_docling.neo4j import Neo4jVector

    Neo4jVector.from_documents.assert_called_once()
    args, kwargs = Neo4jVector.from_documents.call_args
    assert len(args[0]) == 1
    assert args[1] is embedding
    assert kwargs["url"] == "bolt://localhost:7687"
    assert res is neo4j_vs
