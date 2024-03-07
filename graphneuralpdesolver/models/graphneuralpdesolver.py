from typing import Mapping, Any

import jax.numpy as jnp
from flax import linen as nn

from graphneuralpdesolver.graph.typed_graph import (
    TypedGraph, EdgeSet, EdgeSetKey,
    EdgesIndices, NodeSet, Context)
from graphneuralpdesolver.models.deep_typed_graph_net import DeepTypedGraphNet
from graphneuralpdesolver.models.utils import grid_mesh_connectivity_fixed_dx


class AbstractOperator(nn.Module):
  def setup():
    raise NotImplementedError

  def __call__():
    raise NotImplementedError

  @property
  def configs(self):
    configs = {
      attr: self.__getattr__(attr)
      for attr in self.__annotations__.keys() if attr != 'parent'
    }
    return configs

class DummyOperator(AbstractOperator):
  c: float = 0.

  def setup(self):
    self.unused_net = nn.Dense(features=2)

  def __call__(self, specs: jnp.ndarray, u_inp: jnp.ndarray, ndt: float = None):
    a = self.unused_net(u_inp)
    dudt = self.c
    return u_inp + dudt * ndt

class GraphNeuralPDESolver(AbstractOperator):
  """TODO: Add docstrings"""

  domain: Mapping[str, Mapping[str, Any]]
  num_outputs: int = 1
  latent_size: int = 128
  num_mlp_hidden_layers: int = 2  # TRY: 1, 2, 3
  num_message_passing_steps: int = 6  # TRY: tune
  num_gridmesh_cover: int = 4  # TRY: tune
  num_gridmesh_overlap: int = 2  # TRY: tune
  num_multimesh_levels: int = 5  # TRY: tune
  residual_update: bool = True
  time_conditioned: bool = False

  def setup(self):
    # Check the validity of the configurations
    # NOTE: Only fixed dx for now
    self.dx = self.domain['x']['delta']
    self.range_x = self.domain['x']['range']
    self.x = jnp.arange(start=self.range_x[0], stop=self.range_x[1]+self.dx, step=self.dx)
    assert self.x.ndim == 1

    # Initial graphs (holding structural features)
    # NOTE: Only 1D for now  # TODO: Handle 2D cases
    # NOTE: Only fixed dx for now
    # NOTE: Only for periodic BCs for now (circle connections)
    (self.indices_grid, self.indices_mesh), (self.zeta_grid, self.zeta_mesh) =\
      grid_mesh_connectivity_fixed_dx(
        x=self.x, n_cover=self.num_gridmesh_cover, n_overlap=self.num_gridmesh_overlap,
        dx=self.dx, minx=self.range_x[0], maxx=self.range_x[1],
      )
    self._num_grid_nodes = self.zeta_grid.shape[0]
    self._num_mesh_nodes = self.zeta_mesh.shape[0]
    self._grid2mesh_graph = self._init_grid2mesh_graph()
    self._mesh_graph = self._init_mesh_graph()
    self._mesh2grid_graph = self._init_mesh2grid_graph()

    # Encoder, which moves data from the grid to the mesh with a single message
    # passing step.
    self._grid2mesh_gnn = DeepTypedGraphNet(
      embed_nodes=True,  # Embed raw features of the grid and mesh nodes.
      embed_edges=True,  # Embed raw features of the grid2mesh edges.
      edge_latent_size=dict(grid2mesh=self.latent_size),
      node_latent_size=dict(mesh_nodes=self.latent_size, grid_nodes=self.latent_size),
      mlp_hidden_size=self.latent_size,
      mlp_num_hidden_layers=self.num_mlp_hidden_layers,
      num_message_passing_steps=1,
      use_layer_norm=True,
      use_learned_correction=self.time_conditioned,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=True,
      aggregate_normalization=None,
      name='grid2mesh_gnn',
    )

    # Processor, which performs message passing on the multi-mesh.
    self._mesh_gnn = DeepTypedGraphNet(
      embed_nodes=False,  # Node features already embdded by previous layers.
      embed_edges=True,  # Embed raw features of the multi-mesh edges.
      node_latent_size=dict(mesh_nodes=self.latent_size),
      edge_latent_size=dict(mesh=self.latent_size),
      mlp_hidden_size=self.latent_size,
      mlp_num_hidden_layers=self.num_mlp_hidden_layers,
      num_message_passing_steps=self.num_message_passing_steps,
      use_layer_norm=True,
      use_learned_correction=self.time_conditioned,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=False,
      name='mesh_gnn',
    )

    # Decoder, which moves data from the mesh back into the grid with a single
    # message passing step.
    self._mesh2grid_gnn = DeepTypedGraphNet(
      # Require a specific node dimensionaly for the grid node outputs.
      node_output_size=dict(grid_nodes=self.num_outputs),
      embed_nodes=False,  # Node features already embdded by previous layers.
      embed_edges=True,  # Embed raw features of the mesh2grid edges.
      edge_latent_size=dict(mesh2grid=self.latent_size),
      node_latent_size=dict(
          mesh_nodes=self.latent_size,
          grid_nodes=self.latent_size),
      mlp_hidden_size=self.latent_size,
      mlp_num_hidden_layers=self.num_mlp_hidden_layers,
      num_message_passing_steps=1,
      use_layer_norm=True,
      use_learned_correction=self.time_conditioned,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=False,
      name='mesh2grid_gnn',
    )

  def _init_grid2mesh_graph(self) -> TypedGraph:
    """Build Grid2Mesh graph."""

    # CHECK: Try other features
    grid_node_feats = jnp.stack([jnp.abs(self.zeta_grid), jnp.sin(jnp.pi * jnp.abs(self.zeta_grid))], axis=-1)
    mesh_node_feats = jnp.stack([jnp.abs(self.zeta_mesh), jnp.sin(jnp.pi * jnp.abs(self.zeta_mesh))], axis=-1)
    grid_node_set = NodeSet(n_node=jnp.array([self.zeta_grid.shape[0]]), features=grid_node_feats)
    mesh_node_set = NodeSet(n_node=jnp.array([self.zeta_mesh.shape[0]]), features=mesh_node_feats)

    # CHECK: Try other features
    zij = jnp.array([
      jnp.abs(self.zeta_mesh[r]) - jnp.abs(self.zeta_grid[s])
      for s, r in zip(self.indices_grid, self.indices_mesh)
    ])
    edge_feats = jnp.stack([zij], axis=-1)
    edge_set = EdgeSet(
      n_edge=jnp.array([self.indices_grid.shape[0]]),
      indices=EdgesIndices(senders=self.indices_grid, receivers=self.indices_mesh),
      features=edge_feats,
    )

    graph = TypedGraph(
      context=Context(n_graph=jnp.array([1]), features=()),
      nodes={'grid_nodes': grid_node_set, 'mesh_nodes': mesh_node_set},
      edges={EdgeSetKey('grid2mesh', ('grid_nodes', 'mesh_nodes')): edge_set},
    )

    return graph

  def _init_mesh_graph(self) -> TypedGraph:
    """Build Mesh graph."""

    n_nodes_mesh = len(self.zeta_mesh)
    assert n_nodes_mesh > (2 ** self.num_multimesh_levels)
    multimesh_edges = [
      [(idx, ((idx+distance)%n_nodes_mesh)) for idx in range(0, n_nodes_mesh - (distance - 1), distance)]
      for distance in [2 ** p for p in range(self.num_multimesh_levels)]
    ]

    senders = []
    receivers = []
    edge_feats = []
    for edges in multimesh_edges:
      for e in edges:
        # CHECK: Try other features
        senders.append(e[0])
        receivers.append(e[1])
        edge_feats.append([jnp.abs(self.zeta_mesh[e[1]]) - jnp.abs(self.zeta_mesh[e[0]])])
        senders.append(e[1])
        receivers.append(e[0])
        edge_feats.append([jnp.abs(self.zeta_mesh[e[0]]) - jnp.abs(self.zeta_mesh[e[1]])])

    senders = jnp.array(senders)
    receivers = jnp.array(receivers)
    edge_feats = jnp.array(edge_feats)
    edge_set = EdgeSet(
      n_edge=jnp.array([senders.shape[0]]),
      indices=EdgesIndices(senders=senders, receivers=receivers),
      features=edge_feats,
    )

    # CHECK: Try other features
    mesh_node_feats = jnp.stack([jnp.abs(self.zeta_mesh), jnp.sin(jnp.pi * jnp.abs(self.zeta_mesh))], axis=-1)
    mesh_node_set = NodeSet(n_node=jnp.array([self.zeta_mesh.shape[0]]), features=mesh_node_feats)

    graph = TypedGraph(
      context=Context(n_graph=jnp.array([1]), features=()),
      nodes={'mesh_nodes': mesh_node_set},
      edges={EdgeSetKey('mesh', ('mesh_nodes', 'mesh_nodes')): edge_set},
    )

    return graph

  def _init_mesh2grid_graph(self) -> TypedGraph:
    """Build Mesh2Grid graph."""

    # CHECK: Try other features
    grid_node_feats = jnp.stack([jnp.abs(self.zeta_grid), jnp.sin(jnp.pi * jnp.abs(self.zeta_grid))], axis=-1)
    mesh_node_feats = jnp.stack([jnp.abs(self.zeta_mesh), jnp.sin(jnp.pi * jnp.abs(self.zeta_mesh))], axis=-1)
    grid_node_set = NodeSet(n_node=jnp.array([self.zeta_grid.shape[0]]), features=grid_node_feats)
    mesh_node_set = NodeSet(n_node=jnp.array([self.zeta_mesh.shape[0]]), features=mesh_node_feats)

    # CHECK: Try other features
    zij = jnp.array([
      jnp.abs(self.zeta_grid[r]) - jnp.abs(self.zeta_mesh[s])
      for r, s in zip(self.indices_grid, self.indices_mesh)
    ])
    edge_feats = jnp.stack([zij], axis=-1)
    edge_set = EdgeSet(
      n_edge=jnp.array([self.indices_mesh.shape[0]]),
      indices=EdgesIndices(senders=self.indices_mesh, receivers=self.indices_grid),
      features=edge_feats,
    )

    graph = TypedGraph(
      context=Context(n_graph=jnp.array([1]), features=()),
      nodes={'grid_nodes': grid_node_set, 'mesh_nodes': mesh_node_set},
      edges={EdgeSetKey('mesh2grid', ('mesh_nodes', 'grid_nodes')): edge_set},
    )

    return graph

  def __call__(self, specs: jnp.ndarray, u_inp: jnp.ndarray, ndt: float = None):
    assert u_inp.ndim == 4  # [batch_size, 1, num_grid_nodes, num_inputs]
    batch_size = u_inp.shape[0]
    assert u_inp.shape[2] == self.x.shape[0] == self._num_grid_nodes
    if self.residual_update:
      assert u_inp.shape[3] == self.num_outputs
    assert specs.ndim == 2  # [batch_size, num_params]
    assert specs.shape[0] == batch_size

    if self.time_conditioned:
      assert ndt is not None
    else:
      assert ndt is None

    # Prepare the grid node features
    # u -> [num_grid_nodes, batch_size, 1 * num_inputs]
    grid_node_features = jnp.moveaxis(
      u_inp, source=(0, 1, 2), destination=(1, 2, 0)
    ).reshape(self._num_grid_nodes, batch_size, -1)
    # Concatente with equation specifications (parameters, BCs, etc.)
    # CHECK: time is also encoded in Brandstetter
    grid_node_features = jnp.concatenate(axis=-1,
      arrays=[
        jnp.repeat(specs[None, :, :], repeats=self._num_grid_nodes, axis=0),
        grid_node_features,
      ],
    )

    # Transfer data for the grid to the mesh
    # [num_mesh_nodes, batch_size, latent_size], [num_grid_nodes, batch_size, latent_size]
    (latent_mesh_nodes, latent_grid_nodes) = self._run_grid2mesh_gnn(grid_node_features, ndt)

    # Run message passing in the multimesh.
    # [num_mesh_nodes, batch_size, latent_size]
    updated_latent_mesh_nodes = self._run_mesh_gnn(latent_mesh_nodes, ndt)

    # Transfer data frome the mesh to the grid.
    # [num_grid_nodes, batch_size, 1 * num_outputs]
    output_grid_nodes = self._run_mesh2grid_gnn(updated_latent_mesh_nodes, latent_grid_nodes, ndt)

    # Reshape the output to [batch_size, 1, num_grid_nodes, num_outputs]
    output = (output_grid_nodes
      .reshape(self._num_grid_nodes, batch_size, 1, self.num_outputs)
      .swapaxes(0, 1).swapaxes(1, 2)
    )

    # Interpret the output as the first-order derivative
    if self.residual_update:
      du = output  # TRY: du = output * (delta_t * ndt)
      u_out = u_inp + du
    else:
      u_out = output

    return u_out

  def _run_grid2mesh_gnn(self, grid_node_features: jnp.ndarray, ndt: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Runs the grid2mesh_gnn, extracting latent mesh and grid nodes."""

    bsz = grid_node_features.shape[1]
    grid2mesh_graph = self._grid2mesh_graph

    # Concatenate node structural features with input features
    grid_nodes = grid2mesh_graph.nodes['grid_nodes']
    mesh_nodes = grid2mesh_graph.nodes['mesh_nodes']
    new_grid_nodes = grid_nodes._replace(
      features=jnp.concatenate([
        grid_node_features,
        _add_batch_second_axis(
          grid_nodes.features.astype(grid_node_features.dtype),bsz)
      ], axis=-1)
    )
    # To make sure capacity of the embedded is identical for the grid nodes and
    # the mesh nodes, we also append some dummy zero input features for the
    # mesh nodes.
    dummy_mesh_node_features = jnp.zeros(
        (self._num_mesh_nodes,) + grid_node_features.shape[1:],
        dtype=grid_node_features.dtype)
    new_mesh_nodes = mesh_nodes._replace(
      features=jnp.concatenate([
        dummy_mesh_node_features,
        _add_batch_second_axis(
          mesh_nodes.features.astype(dummy_mesh_node_features.dtype), bsz)
      ], axis=-1)
    )

    # Broadcast edge structural features to the required batch size.
    grid2mesh_edges_key = grid2mesh_graph.edge_key_by_name('grid2mesh')
    edges = grid2mesh_graph.edges[grid2mesh_edges_key]
    new_edges = edges._replace(
      features=_add_batch_second_axis(
        edges.features.astype(dummy_mesh_node_features.dtype), bsz)
    )

    input_graph = self._grid2mesh_graph._replace(
      edges={grid2mesh_edges_key: new_edges},
      nodes={
        'grid_nodes': new_grid_nodes,
        'mesh_nodes': new_mesh_nodes
      })

    # Run the GNN.
    grid2mesh_out = self._grid2mesh_gnn(input_graph, correction=ndt)
    latent_mesh_nodes = grid2mesh_out.nodes['mesh_nodes'].features
    latent_grid_nodes = grid2mesh_out.nodes['grid_nodes'].features

    return latent_mesh_nodes, latent_grid_nodes

  def _run_mesh_gnn(self, latent_mesh_nodes: jnp.ndarray, ndt: float) -> jnp.ndarray:
    """Runs the mesh_gnn, extracting updated latent mesh nodes."""

    bsz = latent_mesh_nodes.shape[1]
    mesh_graph = self._mesh_graph

    # Replace the node features
    # CHECK: Try keeping the structural ones
    # NOTE: We don't need to add the structural node features, because these are
    # already part of  the latent state, via the original Grid2Mesh gnn.
    nodes = mesh_graph.nodes['mesh_nodes']
    nodes = nodes._replace(features=latent_mesh_nodes)

    # Add the structural edge features of this graph.
    # NOTE: We need the structural edge features, because it is the first
    # time we are seeing this particular set of edges.
    mesh_edges_key = mesh_graph.edge_key_by_name('mesh')
    # We are assuming here that the mesh gnn uses a single set of edge keys
    # named 'mesh' for the edges and that it uses a single set of nodes named
    # 'mesh_nodes'
    msg = ('The setup currently requires to only have one kind of edge in the mesh GNN.')
    assert len(mesh_graph.edges) == 1, msg
    edges = mesh_graph.edges[mesh_edges_key]
    new_edges = edges._replace(
      features=_add_batch_second_axis(
        edges.features.astype(latent_mesh_nodes.dtype), bsz)
    )

    # Build the graph
    input_graph = mesh_graph._replace(
      edges={mesh_edges_key: new_edges}, nodes={'mesh_nodes': nodes}
    )

    # Run the GNN
    return self._mesh_gnn(input_graph, correction=ndt).nodes['mesh_nodes'].features

  def _run_mesh2grid_gnn(self, updated_latent_mesh_nodes: jnp.ndarray,
                        latent_grid_nodes: jnp.ndarray, ndt: float) -> jnp.ndarray:
    """Runs the mesh2grid_gnn, extracting the output grid nodes."""

    bsz = updated_latent_mesh_nodes.shape[1]
    mesh2grid_graph = self._mesh2grid_graph

    # NOTE: We don't need to add the structural node features, because these are
    # already part of the latent state, via the original Grid2Mesh gnn.
    mesh_nodes = mesh2grid_graph.nodes['mesh_nodes']
    grid_nodes = mesh2grid_graph.nodes['grid_nodes']
    new_mesh_nodes = mesh_nodes._replace(features=updated_latent_mesh_nodes)
    new_grid_nodes = grid_nodes._replace(features=latent_grid_nodes)

    # Add the structural edge features of this graph.
    # NOTE: We need the structural edge features, because it is the first time we
    # are seeing this particular set of edges.
    mesh2grid_key = mesh2grid_graph.edge_key_by_name('mesh2grid')
    edges = mesh2grid_graph.edges[mesh2grid_key]
    new_edges = edges._replace(
      features=_add_batch_second_axis(
        edges.features.astype(latent_grid_nodes.dtype), bsz))

    # Build the new graph
    input_graph = mesh2grid_graph._replace(
      edges={mesh2grid_key: new_edges},
      nodes={
        'mesh_nodes': new_mesh_nodes,
        'grid_nodes': new_grid_nodes
      })

    # Run the GNN
    output_graph = self._mesh2grid_gnn(input_graph, correction=ndt)
    output_grid_nodes = output_graph.nodes['grid_nodes'].features

    return output_grid_nodes

def _add_batch_second_axis(data, bsz):
  # data [leading_dim, trailing_dim]
  assert data.ndim == 2
  ones = jnp.ones([bsz, 1], dtype=data.dtype)
  return data[:, None] * ones  # [leading_dim, batch, trailing_dim]
