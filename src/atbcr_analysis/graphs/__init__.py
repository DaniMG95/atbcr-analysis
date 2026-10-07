"""Graph registry and built-in graph generators."""

from atbcr_analysis.graphs.barabasi_albert import build_barabasi_albert_graph
from atbcr_analysis.graphs.base import Adjacency, Edge, GraphBuilder, GraphGeneratorConfig
from atbcr_analysis.graphs.complete import build_complete_graph
from atbcr_analysis.graphs.erdos_renyi import build_erdos_renyi_graph
from atbcr_analysis.graphs.factory import GraphFactory, generate_graph, register_graph
from atbcr_analysis.graphs.ring import build_ring_graph
from atbcr_analysis.graphs.utils import graph_edges
from atbcr_analysis.graphs.watts_strogatz import build_watts_strogatz_graph

__all__ = [
    "Adjacency",
    "Edge",
    "GraphBuilder",
    "GraphFactory",
    "GraphGeneratorConfig",
    "build_barabasi_albert_graph",
    "build_complete_graph",
    "build_erdos_renyi_graph",
    "build_ring_graph",
    "build_watts_strogatz_graph",
    "generate_graph",
    "graph_edges",
    "register_graph",
]
