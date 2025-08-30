"""Utilities for ingesting Docling processed PDFs into Neo4j."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Union

from docling.chunking import BaseChunker, HybridChunker
from langchain_community.vectorstores import Neo4jVector
from langchain_core.embeddings import Embeddings

from langchain_docling.loader import DoclingLoader


def ingest_pdfs_to_neo4j(
    file_paths: Union[str, Iterable[str]],
    *,
    embedding: Embeddings,
    url: str,
    username: str,
    password: str,
    convert_kwargs: Optional[dict[str, Any]] = None,
    chunker: Optional[BaseChunker] = None,
    database: Optional[str] = None,
    **neo4j_kwargs: Any,
) -> Neo4jVector:
    """Ingest PDF files into a Neo4j vector store using Docling's hybrid chunker.

    Args:
        file_paths: Path(s) or URLs to PDF files to ingest.
        embedding: Embedding model used to compute vector representations of the
            chunks.
        url: Neo4j connection URI, e.g. ``bolt://localhost:7687``.
        username: Neo4j authentication username.
        password: Neo4j authentication password.
        convert_kwargs: Optional kwargs forwarded to Docling's converter.
        chunker: Optional Docling chunker instance to use. Defaults to
            :class:`HybridChunker`.
        database: Optional Neo4j database name.
        **neo4j_kwargs: Additional keyword arguments forwarded to
            ``Neo4jVector.from_documents``.

    Returns:
        The created :class:`Neo4jVector` instance containing the ingested
        documents.

    Example:
        .. code-block:: python

            from langchain_openai import OpenAIEmbeddings
            from langchain_docling.neo4j import ingest_pdfs_to_neo4j

            vector_store = ingest_pdfs_to_neo4j(
                file_paths=["/path/to/file.pdf"],
                embedding=OpenAIEmbeddings(),
                url="bolt://localhost:7687",
                username="neo4j",
                password="secret",
            )
    """
    loader = DoclingLoader(
        file_path=file_paths,
        convert_kwargs=convert_kwargs,
        chunker=chunker or HybridChunker(),
    )
    documents = list(loader.lazy_load())
    vector_store = Neo4jVector.from_documents(
        documents=documents,
        embedding=embedding,
        url=url,
        username=username,
        password=password,
        database=database,
        **neo4j_kwargs,
    )
    return vector_store
