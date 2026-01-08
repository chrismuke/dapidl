"""Cell Ontology OBO File Loader.

This module handles loading the full Cell Ontology from OBO format files,
providing access to all ~2,800 CL terms with their complete hierarchy.

The OBO file can be downloaded from:
    https://raw.githubusercontent.com/obophenotype/cell-ontology/master/cl.obo

For DAPIDL, we primarily use the curated subset in cl_database.py, but this
loader enables:
1. Dynamic validation of CL IDs
2. Full ancestor traversal
3. Semantic similarity via hierarchy distance
4. Fuzzy matching via synonyms

Usage:
    from dapidl.ontology.cl_loader import CLLoader

    loader = CLLoader()
    loader.load()  # Downloads if not cached

    # Look up term
    term = loader.get_term("CL:0000624")
    ancestors = loader.get_ancestors("CL:0000624")
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterator, Optional

import networkx as nx
from loguru import logger

# Cache directory for downloaded OBO files
CACHE_DIR = Path(tempfile.gettempdir()) / "dapidl_ontology_cache"
CL_OBO_URL = "https://raw.githubusercontent.com/obophenotype/cell-ontology/master/cl.obo"


class CLLoader:
    """Loader for Cell Ontology from OBO format.

    Provides lazy loading and caching of the full Cell Ontology graph.
    Uses obonet for parsing and networkx for hierarchy traversal.
    """

    def __init__(self, cache_dir: Path | str | None = None):
        """Initialize the loader.

        Args:
            cache_dir: Directory for caching downloaded OBO files.
                      Defaults to system temp directory.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._graph: Optional[nx.MultiDiGraph] = None
        self._name_to_id: dict[str, str] = {}
        self._synonym_to_id: dict[str, list[str]] = {}

    @property
    def obo_path(self) -> Path:
        """Path to cached OBO file."""
        return self.cache_dir / "cl.obo"

    @property
    def is_loaded(self) -> bool:
        """Check if ontology is loaded."""
        return self._graph is not None

    def download(self, force: bool = False) -> Path:
        """Download the Cell Ontology OBO file.

        Args:
            force: If True, re-download even if cached.

        Returns:
            Path to the downloaded file.
        """
        if self.obo_path.exists() and not force:
            logger.debug(f"Using cached OBO file: {self.obo_path}")
            return self.obo_path

        logger.info(f"Downloading Cell Ontology from {CL_OBO_URL}...")

        import urllib.request

        urllib.request.urlretrieve(CL_OBO_URL, self.obo_path)
        logger.info(f"Downloaded to {self.obo_path}")

        return self.obo_path

    def load(self, obo_path: Path | str | None = None) -> None:
        """Load the Cell Ontology from OBO file.

        Args:
            obo_path: Path to OBO file. Downloads if not provided.
        """
        if obo_path is None:
            obo_path = self.download()
        else:
            obo_path = Path(obo_path)

        if not obo_path.exists():
            raise FileNotFoundError(f"OBO file not found: {obo_path}")

        logger.info(f"Loading Cell Ontology from {obo_path}...")

        import obonet

        self._graph = obonet.read_obo(obo_path)
        logger.info(f"Loaded {len(self._graph)} terms")

        # Build lookup indices
        self._build_indices()

    def _build_indices(self) -> None:
        """Build name and synonym lookup indices."""
        self._name_to_id.clear()
        self._synonym_to_id.clear()

        for node_id, data in self._graph.nodes(data=True):
            if not node_id.startswith("CL:"):
                continue

            # Index by canonical name
            name = data.get("name", "")
            if name:
                self._name_to_id[name.lower()] = node_id

            # Index by synonyms
            synonyms = data.get("synonym", [])
            if isinstance(synonyms, str):
                synonyms = [synonyms]

            for syn in synonyms:
                # Extract synonym text from OBO format
                # Format: "synonym text" EXACT/RELATED/BROAD [source]
                if '"' in syn:
                    syn_text = syn.split('"')[1].lower()
                    if syn_text not in self._synonym_to_id:
                        self._synonym_to_id[syn_text] = []
                    self._synonym_to_id[syn_text].append(node_id)

        logger.debug(
            f"Built indices: {len(self._name_to_id)} names, "
            f"{len(self._synonym_to_id)} synonyms"
        )

    def ensure_loaded(self) -> None:
        """Ensure ontology is loaded, loading if necessary."""
        if not self.is_loaded:
            self.load()

    def get_term_data(self, cl_id: str) -> dict | None:
        """Get raw term data from the ontology graph.

        Args:
            cl_id: CL ID like "CL:0000624"

        Returns:
            Node data dict or None if not found
        """
        self.ensure_loaded()
        if cl_id in self._graph:
            return dict(self._graph.nodes[cl_id])
        return None

    def get_name(self, cl_id: str) -> str | None:
        """Get the canonical name for a CL ID."""
        data = self.get_term_data(cl_id)
        return data.get("name") if data else None

    def get_synonyms(self, cl_id: str) -> list[str]:
        """Get all synonyms for a CL ID."""
        data = self.get_term_data(cl_id)
        if not data:
            return []

        synonyms = data.get("synonym", [])
        if isinstance(synonyms, str):
            synonyms = [synonyms]

        # Extract synonym text
        result = []
        for syn in synonyms:
            if '"' in syn:
                result.append(syn.split('"')[1])
        return result

    def get_definition(self, cl_id: str) -> str | None:
        """Get the definition for a CL ID."""
        data = self.get_term_data(cl_id)
        if not data:
            return None

        defn = data.get("def")
        if defn and '"' in defn:
            return defn.split('"')[1]
        return defn

    def find_by_name(self, name: str) -> str | None:
        """Find CL ID by canonical name (case-insensitive).

        Args:
            name: Cell type name

        Returns:
            CL ID or None if not found
        """
        self.ensure_loaded()
        return self._name_to_id.get(name.lower())

    def find_by_synonym(self, synonym: str) -> list[str]:
        """Find CL IDs by synonym (case-insensitive).

        Args:
            synonym: Synonym text

        Returns:
            List of matching CL IDs (may be empty)
        """
        self.ensure_loaded()
        return self._synonym_to_id.get(synonym.lower(), [])

    def get_parents(self, cl_id: str) -> list[str]:
        """Get direct parent CL IDs (via is_a relationship).

        Args:
            cl_id: CL ID

        Returns:
            List of parent CL IDs
        """
        self.ensure_loaded()

        if cl_id not in self._graph:
            return []

        parents = []
        # In obonet, edges go from child to parent
        for _, parent, edge_data in self._graph.out_edges(cl_id, data=True):
            if edge_data.get("relationship") in (None, "is_a"):
                if parent.startswith("CL:"):
                    parents.append(parent)
        return parents

    def get_children(self, cl_id: str) -> list[str]:
        """Get direct child CL IDs.

        Args:
            cl_id: CL ID

        Returns:
            List of child CL IDs
        """
        self.ensure_loaded()

        if cl_id not in self._graph:
            return []

        children = []
        for child, _, edge_data in self._graph.in_edges(cl_id, data=True):
            if edge_data.get("relationship") in (None, "is_a"):
                if child.startswith("CL:"):
                    children.append(child)
        return children

    def get_ancestors(self, cl_id: str, include_self: bool = False) -> list[str]:
        """Get all ancestor CL IDs (transitive closure of parents).

        Args:
            cl_id: CL ID
            include_self: Whether to include the term itself

        Returns:
            List of ancestor CL IDs (closest first)
        """
        self.ensure_loaded()

        if cl_id not in self._graph:
            return [cl_id] if include_self else []

        ancestors = []
        if include_self:
            ancestors.append(cl_id)

        # BFS traversal up the hierarchy
        visited = {cl_id}
        queue = [cl_id]

        while queue:
            current = queue.pop(0)
            parents = self.get_parents(current)

            for parent in parents:
                if parent not in visited:
                    visited.add(parent)
                    ancestors.append(parent)
                    queue.append(parent)

        return ancestors

    def get_descendants(self, cl_id: str, include_self: bool = False) -> list[str]:
        """Get all descendant CL IDs (transitive closure of children).

        Args:
            cl_id: CL ID
            include_self: Whether to include the term itself

        Returns:
            List of descendant CL IDs
        """
        self.ensure_loaded()

        if cl_id not in self._graph:
            return [cl_id] if include_self else []

        descendants = []
        if include_self:
            descendants.append(cl_id)

        # BFS traversal down the hierarchy
        visited = {cl_id}
        queue = [cl_id]

        while queue:
            current = queue.pop(0)
            children = self.get_children(current)

            for child in children:
                if child not in visited:
                    visited.add(child)
                    descendants.append(child)
                    queue.append(child)

        return descendants

    def get_distance(self, cl_id1: str, cl_id2: str) -> int | None:
        """Get shortest path distance between two CL IDs.

        Uses undirected path through the hierarchy.

        Args:
            cl_id1: First CL ID
            cl_id2: Second CL ID

        Returns:
            Shortest path length or None if no path exists
        """
        self.ensure_loaded()

        if cl_id1 not in self._graph or cl_id2 not in self._graph:
            return None

        try:
            # Convert to undirected for path finding
            undirected = self._graph.to_undirected()
            return nx.shortest_path_length(undirected, cl_id1, cl_id2)
        except nx.NetworkXNoPath:
            return None

    def get_lowest_common_ancestor(self, cl_id1: str, cl_id2: str) -> str | None:
        """Find the lowest common ancestor of two CL IDs.

        Args:
            cl_id1: First CL ID
            cl_id2: Second CL ID

        Returns:
            LCA CL ID or None if none exists
        """
        self.ensure_loaded()

        ancestors1 = set(self.get_ancestors(cl_id1, include_self=True))
        ancestors2 = self.get_ancestors(cl_id2, include_self=True)

        # Find first common ancestor (closest to both)
        for anc in ancestors2:
            if anc in ancestors1:
                return anc

        return None

    def iter_terms(self) -> Iterator[tuple[str, dict]]:
        """Iterate over all CL terms.

        Yields:
            Tuples of (cl_id, node_data)
        """
        self.ensure_loaded()

        for node_id, data in self._graph.nodes(data=True):
            if node_id.startswith("CL:"):
                yield node_id, data

    def count_terms(self) -> int:
        """Count total number of CL terms."""
        self.ensure_loaded()
        return sum(1 for node_id in self._graph.nodes if node_id.startswith("CL:"))


# Singleton instance for convenience
_default_loader: Optional[CLLoader] = None


def get_loader() -> CLLoader:
    """Get the default CLLoader instance (singleton)."""
    global _default_loader
    if _default_loader is None:
        _default_loader = CLLoader()
    return _default_loader


def ensure_ontology_loaded() -> CLLoader:
    """Ensure the default ontology is loaded and return the loader."""
    loader = get_loader()
    loader.ensure_loaded()
    return loader


if __name__ == "__main__":
    # Test loading
    loader = CLLoader()
    loader.load()

    print(f"Total CL terms: {loader.count_terms()}")

    # Test some lookups
    test_ids = ["CL:0000624", "CL:0000625", "CL:0000084", "CL:0000066"]

    for cl_id in test_ids:
        name = loader.get_name(cl_id)
        parents = loader.get_parents(cl_id)
        ancestors = loader.get_ancestors(cl_id)
        print(f"\n{cl_id}: {name}")
        print(f"  Parents: {[loader.get_name(p) for p in parents]}")
        print(f"  Ancestors ({len(ancestors)}): {ancestors[:5]}...")

    # Test LCA
    lca = loader.get_lowest_common_ancestor("CL:0000624", "CL:0000625")
    print(f"\nLCA of CD4+ T and CD8+ T: {lca} ({loader.get_name(lca)})")
