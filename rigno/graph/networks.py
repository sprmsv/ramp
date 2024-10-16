# Adopted from https://github.com/google-deepmind/graphcast
# Accessed on 16 February 2024, commit 8debd7289bb2c498485f79dbd98d8b4933bfc6a7
# Codes are modified
"""A library of typed Graph Neural Networks."""

from typing import Callable, Mapping, Optional, Union, Tuple

import jax.numpy as jnp
import jax.tree_util as tree

from rigno.graph.entities import TypedGraph, EdgeSetKey, EdgeSet, NodeSet, Context
from rigno.models.utils import mean_aggregation_edge
from rigno.utils import Array

# All features will be an ArrayTree
NodeFeatures = EdgeFeatures = SenderFeatures = ReceiverFeatures = EdgeMask = Globals = Array

EmbedEdgeFn = Callable[
  [EdgeFeatures],
  EdgeFeatures
]
EmbedNodeFn = Callable[
  [NodeFeatures],
  NodeFeatures
]
EmbedGlobalFn = Callable[
  [Globals],
  Globals
]
GNUpdateEdgeFn = Callable[
  [EdgeFeatures, Mapping[str, ReceiverFeatures], Globals],
  EdgeFeatures
]
GNUpdateNodeFn = Callable[
  [NodeFeatures, Mapping[str, ReceiverFeatures], Globals],
  NodeFeatures
]
GNUpdateGlobalFn = Callable[
  [Mapping[str, NodeFeatures], Mapping[str, EdgeFeatures], Globals],
  Globals
]
InteractionUpdateEdgeFn = Callable[
  [EdgeFeatures, Mapping[str, ReceiverFeatures]],
  EdgeFeatures
]
InteractionUpdateNodeFn = Callable[
  [NodeFeatures, Mapping[str, ReceiverFeatures]],
  NodeFeatures
]

AggregateEdgesToNodesFn = Callable[
  [ReceiverFeatures, EdgeMask, Tuple],
  NodeFeatures
]
AggregateNodesToGlobalsFn = Callable[
  [NodeFeatures, Tuple],
  Globals
]
AggregateEdgesToGlobalsFn = Callable[
  [ReceiverFeatures, EdgeMask, Tuple],
  Globals
]

def GraphNetwork(
    update_edge_fn: Mapping[str, GNUpdateEdgeFn],
    update_node_fn: Mapping[str, GNUpdateNodeFn],
    update_global_fn: Optional[GNUpdateGlobalFn] = None,
    aggregate_edges_for_nodes_fn: AggregateEdgesToNodesFn = mean_aggregation_edge,
    aggregate_nodes_for_globals_fn: AggregateNodesToGlobalsFn = jnp.mean,
    aggregate_edges_for_globals_fn: AggregateEdgesToGlobalsFn = mean_aggregation_edge,
  ):
  """Returns a method that applies a configured GraphNetwork.

  This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261
  extended to Typed Graphs with multiple edge sets and node sets and extended to
  allow aggregating not only edges received by the nodes, but also edges sent by
  the nodes.

  Example usage::

    gn = GraphNetwork(update_edge_function,
    update_node_function, **kwargs)
    # Conduct multiple rounds of message passing with the same parameters:
    for _ in range(num_message_passing_steps):
      graph = gn(graph)

  Args:
    update_edge_fn: mapping of functions used to update a subset of the edge
      types, indexed by edge type name.
    update_node_fn: mapping of functions used to update a subset of the node
      types, indexed by node type name.
    update_global_fn: function used to update the globals or None to deactivate
      globals updates.
    aggregate_edges_for_nodes_fn: function used to aggregate messages to each
      node.
    aggregate_nodes_for_globals_fn: function used to aggregate the nodes for the
      globals.
    aggregate_edges_for_globals_fn: function used to aggregate the edges for the
      globals.

  Returns:
    A method that applies the configured GraphNetwork.
  """

  def _apply_graph_net(graph: TypedGraph, **kwargs) -> TypedGraph:
    """Applies a configured GraphNetwork to a graph.

    This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261
    extended to Typed Graphs with multiple edge sets and node sets and extended
    to allow aggregating not only edges received by the nodes, but also edges
    sent by the nodes.

    Args:
      graph: a `TypedGraph` containing the graph.

    Returns:
      Updated `TypedGraph`.
    """

    updated_graph = graph

    # Update each edge set using its corresponding update function
    updated_edges = dict(updated_graph.edges)
    for edge_set_name, edge_fn in update_edge_fn.items():
      edge_set_key = graph.edge_key_by_name(edge_set_name)
      updated_edges[edge_set_key] = _edge_update(updated_graph, edge_fn, edge_set_key, kwargs)
    updated_graph = updated_graph._replace(edges=updated_edges)

    # Update each node set using its corresponding update function
    updated_nodes = dict(updated_graph.nodes)
    for node_set_key, node_fn in update_node_fn.items():
      updated_nodes[node_set_key] = _node_update(
          updated_graph, node_fn, node_set_key, aggregate_edges_for_nodes_fn, kwargs)
    updated_graph = updated_graph._replace(nodes=updated_nodes)

    # Update global features
    if update_global_fn:
      updated_context = _global_update(
        updated_graph,
        update_global_fn,
        aggregate_edges_for_globals_fn,
        aggregate_nodes_for_globals_fn,
        kwargs,
      )
      updated_graph = updated_graph._replace(context=updated_context)

    return updated_graph

  return _apply_graph_net

def _edge_update(graph: TypedGraph, edge_fn: GNUpdateEdgeFn,
  edge_set_key: EdgeSetKey, fn_kwargs: dict) -> EdgeSet:
  """Updates an edge set of a given key."""

  # Get sender and receivers
  receiver_nodes = graph.nodes[edge_set_key.node_sets[1]]
  edge_set = graph.edges[edge_set_key]

  # Get received features
  received_attributes = jnp.tile(
    jnp.expand_dims(receiver_nodes.features, axis=2),
    reps=(1, 1, edge_set.features.shape[2], 1)
  )

  # Get new edge features
  global_features = tree.tree_map(
    lambda g: jnp.tile(
      jnp.expand_dims(g, axis=(1, 2)),
      reps=(1, edge_set.features.shape[1], edge_set.features.shape[2], 1),
    ),
    graph.context.features
  )
  new_features = edge_fn(edge_set.features, received_attributes, global_features, **fn_kwargs)

  return edge_set._replace(features=new_features)

def _node_update(graph: TypedGraph, node_fn: GNUpdateNodeFn,
  node_set_key: str, aggregation_fn: Callable, fn_kwargs: dict) -> NodeSet:
  """Updates an edge set of a given key."""

  # Get node set and its shape
  node_set = graph.nodes[node_set_key]
  sum_n_node = tree.tree_leaves(node_set.features)[0].shape[1]

  # Aggregate received features
  received_features = {}
  for edge_set_key, edge_set in graph.edges.items():
    receiver_node_set_key = edge_set_key.node_sets[1]
    if receiver_node_set_key == node_set_key:
      received_features[edge_set_key.name] = aggregation_fn(edge_set.features, edge_set.mask, axis=2)

  # Get new node features
  n_node = node_set.n_node[0]
  global_features = tree.tree_map(
    lambda g: jnp.repeat(g, n_node, axis=1, total_repeat_length=sum_n_node),
    graph.context.features,
  )
  new_features = node_fn(node_set.features, received_features, global_features, **fn_kwargs)

  return node_set._replace(features=new_features)

def _global_update(graph: TypedGraph, global_fn: GNUpdateGlobalFn,
  edge_aggregation_fn: AggregateEdgesToGlobalsFn,
  node_aggregation_fn: AggregateNodesToGlobalsFn, fn_kwargs: dict) -> Context:
  """Updates an edge set of a given key."""

  # Aggregate edge features
  edge_features = {}
  for edge_set_key, edge_set in graph.edges.items():
    edge_features[edge_set_key.name] = edge_aggregation_fn(edge_set.features, edge_set.mask, axis=(1, 2))

  # Aggregate node features
  node_features = {}
  for node_set_key, node_set in graph.nodes.items():
    node_features[node_set_key] = node_aggregation_fn(node_set.features, axis=1)

  # Get new global features
  new_features = global_fn(node_features, edge_features, graph.context.features, **fn_kwargs)

  return graph.context._replace(features=new_features)

def InteractionNetwork(
    update_edge_fn: Mapping[str, InteractionUpdateEdgeFn],
    update_node_fn: Mapping[str, InteractionUpdateNodeFn],
    aggregate_edges_for_nodes_fn: AggregateEdgesToNodesFn = mean_aggregation_edge,
  ):
  """Returns a method that applies a configured InteractionNetwork.

  An interaction network computes interactions on the edges based on the
  previous edges features, and on the features of the nodes sending into those
  edges. It then updates the nodes based on the incoming updated edges.
  See https://arxiv.org/abs/1612.00222 for more details.

  This implementation extends the behavior to `TypedGraphs` adding an option
  to include edge features for which a node is a sender in the arguments to
  the node update function.

  Args:
    update_edge_fn: mapping of functions used to update a subset of the edge
      types, indexed by edge type name.
    update_node_fn: mapping of functions used to update a subset of the node
      types, indexed by node type name.
    aggregate_edges_for_nodes_fn: function used to aggregate messages to each
      node.
  """
  # An InteractionNetwork is a GraphNetwork without globals features,
  # so we implement the InteractionNetwork as a configured GraphNetwork.

  # TODO: TRY: Consider using the global features instead of keyword arguments

  # An InteractionNetwork edge function does not have global feature inputs,
  # so we filter the passed global argument in the GraphNetwork.
  wrapped_update_edge_fn = tree.tree_map(lambda fn: lambda e, r, g, **kw: fn(e, r, **kw), update_edge_fn)

  # Similarly, we wrap the update_node_fn to ensure only the expected
  # arguments are passed to the Interaction net.
  wrapped_update_node_fn = tree.tree_map(lambda fn: lambda n, r, g, **kw: fn(n, r, **kw), update_node_fn)

  # We build a graph network using the wrapped update functions
  interaction_network = GraphNetwork(
    update_edge_fn=wrapped_update_edge_fn,
    update_node_fn=wrapped_update_node_fn,
    aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn,
  )

  return interaction_network

def GraphMapFeatures(
    embed_edge_fn: Optional[Mapping[str, EmbedEdgeFn]] = None,
    embed_node_fn: Optional[Mapping[str, EmbedNodeFn]] = None,
    embed_global_fn: Optional[EmbedGlobalFn] = None,
  ):
  """Returns function which embeds the components of a graph independently.

  Args:
    embed_edge_fn: mapping of functions used to embed each edge type,
      indexed by edge type name.
    embed_node_fn: mapping of functions used to embed each node type,
      indexed by node type name.
    embed_global_fn: function used to embed the globals.
  """

  def _embed(graph: TypedGraph, **kwargs) -> TypedGraph:

    updated_edges = dict(graph.edges)
    if embed_edge_fn:
      for edge_set_name, embed_fn in embed_edge_fn.items():
        edge_set_key = graph.edge_key_by_name(edge_set_name)
        edge_set = graph.edges[edge_set_key]
        updated_edges[edge_set_key] = edge_set._replace(features=embed_fn(edge_set.features, **kwargs))

    updated_nodes = dict(graph.nodes)
    if embed_node_fn:
      for node_set_key, embed_fn in embed_node_fn.items():
        node_set = graph.nodes[node_set_key]
        updated_nodes[node_set_key] = node_set._replace(
          features=embed_fn(node_set.features, **kwargs))

    updated_context = graph.context
    if embed_global_fn:
      updated_context = updated_context._replace(
        features=embed_global_fn(updated_context.features, **kwargs))

    return graph._replace(edges=updated_edges, nodes=updated_nodes, context=updated_context)

  return _embed
