from typing import Sequence, Tuple, Union

import numpy as np
import jax.numpy as jnp
import jax.random
from flax import linen as nn
import flax.typing

from mpgno.graph.typed_graph import (
    TypedGraph, EdgeSet, EdgeSetKey,
    EdgesIndices, NodeSet, Context)
from mpgno.models.deep_typed_graph_net import DeepTypedGraphNet
from mpgno.models.utils import compute_derivatives
from mpgno.utils import Array, normalize, shuffle_arrays


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

class MPGNO(AbstractOperator):
  """TODO: Add docstrings"""
  # NOTE: Only fixed dx is supported for now

  num_outputs: int
  num_grid_nodes: Sequence[int]
  num_mesh_nodes: Sequence[int]
  use_t: bool = True
  use_tau: bool = True
  conditional_normalization: bool = False
  deriv_degree: int = 0
  latent_size: int = 128
  num_mlp_hidden_layers: int = 2
  num_message_passing_steps: int = 18
  num_message_passing_steps_grid: int = 2
  overlap_factor_grid2mesh: float = 1.
  overlap_factor_mesh2grid: float = 1.
  num_multimesh_levels: int = 1
  node_coordinate_freqs: int = 1
  p_dropout_edges_grid2mesh: int = 0.
  p_dropout_edges_mesh2grid: int = 0.

  def setup(self):

    # TODO: Add upper bounds
    assert self.overlap_factor_grid2mesh >= 1.0
    assert self.overlap_factor_mesh2grid >= 1.0

    # Initialize the structural features of the grid points
    self.zeta_grid = np.stack(
      arrays=np.meshgrid(
        (2 * (np.arange(self.num_grid_nodes[0]) / self.num_grid_nodes[0]) - 1),
        (2 * (np.arange(self.num_grid_nodes[1]) / self.num_grid_nodes[1]) - 1),
      ),
      axis=-1,
    ).swapaxes(0, 1)
    self.dz_grid = 2 / np.array(self.num_grid_nodes)
    # Initialize the structural features of the mesh points
    self.zeta_mesh = np.stack(
      arrays=np.meshgrid(
        (2 * (np.arange(self.num_mesh_nodes[0]) / self.num_mesh_nodes[0]) - 1),
        (2 * (np.arange(self.num_mesh_nodes[1]) / self.num_mesh_nodes[1]) - 1),
      ),
      axis=-1,
    ).swapaxes(0, 1)
    self.dz_mesh = 2 / np.array(self.num_mesh_nodes)

    self._num_grid_nodes_tot = np.prod(self.num_grid_nodes).item()
    self._num_mesh_nodes_tot = np.prod(self.num_mesh_nodes).item()

    # Get grid2mesh edge connections
    _idx_grid2mesh_grid_nodes: list[int] = []
    _idx_grid2mesh_mesh_nodes: list[int] = []
    for idx_mesh_node_flat in range(self._num_mesh_nodes_tot):
      idx_edges_grid, idx_edges_mesh = self._get_connections_by_mesh_node(
        idx_mesh_node_flat, overlap_factor=self.overlap_factor_grid2mesh, ord_distance=2)
      _idx_grid2mesh_grid_nodes.append(idx_edges_grid)  # flat index
      _idx_grid2mesh_mesh_nodes.append(idx_edges_mesh)  # flat index
    self.idx_grid2mesh_grid_nodes = np.concatenate(_idx_grid2mesh_grid_nodes).tolist()
    self.idx_grid2mesh_mesh_nodes = np.concatenate(_idx_grid2mesh_mesh_nodes).tolist()

    # Get multimesh edge connections
    # TODO: Optimize the procedure (avoid for loops)
    assert (2 ** self.num_multimesh_levels) < max(self.num_mesh_nodes)
    idx_senders = []
    idx_receivers = []
    for p in range(self.num_multimesh_levels):
      dist = (
        min(2 ** p, self.num_mesh_nodes[0] // 2),
        min(2 ** p, self.num_mesh_nodes[1] // 2)
      )
      for i in np.arange(self.num_mesh_nodes[0], step=dist[0]):
        for j in np.arange(self.num_mesh_nodes[1], step=dist[1]):
          idx_senders.append((i, j))  # LEFT
          idx_receivers.append(((i-dist[0]) % self.num_mesh_nodes[0], j))
          idx_senders.append((i, j))  # RIGHT
          idx_receivers.append(((i+dist[0]) % self.num_mesh_nodes[0], j))
          idx_senders.append((i, j))  # DOWN
          idx_receivers.append((i, (j-dist[1]) % self.num_mesh_nodes[0]))
          idx_senders.append((i, j))  # UP
          idx_receivers.append((i, (j+dist[1]) % self.num_mesh_nodes[0]))
    # Get flat indexes
    self.idx_multimesh_send = list(
      map(lambda s: s[0] * self.num_mesh_nodes[0] + s[1], idx_senders))
    self.idx_multimesh_recv = list(
      map(lambda r: r[0] * self.num_mesh_nodes[0] + r[1], idx_receivers))

    # Get mesh2grid edge connections
    _idx_mesh2grid_grid_nodes: list[int] = []
    _idx_mesh2grid_mesh_nodes: list[int] = []
    for idx_mesh_node_flat in range(self._num_mesh_nodes_tot):
      idx_edges_grid, idx_edges_mesh = self._get_connections_by_mesh_node(
        idx_mesh_node_flat, overlap_factor=self.overlap_factor_mesh2grid, ord_distance=2)
      _idx_mesh2grid_grid_nodes.append(idx_edges_grid)  # flat index
      _idx_mesh2grid_mesh_nodes.append(idx_edges_mesh)  # flat index
    self.idx_mesh2grid_grid_nodes = np.concatenate(_idx_mesh2grid_grid_nodes).tolist()
    self.idx_mesh2grid_mesh_nodes = np.concatenate(_idx_mesh2grid_mesh_nodes).tolist()

    # Get grid2grid edge connections
    # TODO: Optimize the procedure (avoid for loops)
    idx_senders = []
    idx_receivers = []
    for p in range(1):
      dist = (
        min(2 ** p, self.num_grid_nodes[0] // 2),
        min(2 ** p, self.num_grid_nodes[1] // 2)
      )
      for i in np.arange(self.num_grid_nodes[0], step=dist[0]):
        for j in np.arange(self.num_grid_nodes[1], step=dist[1]):
          idx_senders.append((i, j))  # LEFT
          idx_receivers.append(((i-dist[0]) % self.num_grid_nodes[0], j))
          idx_senders.append((i, j))  # RIGHT
          idx_receivers.append(((i+dist[0]) % self.num_grid_nodes[0], j))
          idx_senders.append((i, j))  # DOWN
          idx_receivers.append((i, (j-dist[1]) % self.num_grid_nodes[0]))
          idx_senders.append((i, j))  # UP
          idx_receivers.append((i, (j+dist[1]) % self.num_grid_nodes[0]))
    # Get flat indexes
    self.idx_grid2grid_send = list(
      map(lambda s: s[0] * self.num_grid_nodes[0] + s[1], idx_senders))
    self.idx_grid2grid_recv = list(
      map(lambda r: r[0] * self.num_grid_nodes[0] + r[1], idx_receivers))

    # Initialize the graphs
    self._grid2mesh_graph = self._init_grid2mesh_graph()
    self._mesh_graph = self._init_mesh_graph()
    self._mesh2grid_graph = self._init_mesh2grid_graph()
    self._grid2grid_graph = self._init_grid2grid_graph()

    # Define the encoder
    self._grid2mesh_gnn = DeepTypedGraphNet(
      embed_nodes=True,  # Embed raw features of the grid and mesh nodes.
      embed_edges=True,  # Embed raw features of the grid2mesh edges.
      edge_latent_size=dict(grid2mesh=self.latent_size),
      node_latent_size=dict(mesh_nodes=self.latent_size, grid_nodes=self.latent_size),
      mlp_hidden_size=self.latent_size,
      mlp_num_hidden_layers=self.num_mlp_hidden_layers,
      num_message_passing_steps=1,
      use_layer_norm=True,
      conditional_normalization=self.conditional_normalization,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=True,
      aggregate_edges_for_nodes_fn='segment_mean',
      aggregate_normalization=None,
      name='grid2mesh_gnn',
    )

    # Define the processor
    self._mesh_gnn = DeepTypedGraphNet(
      embed_nodes=False,  # Node features already embdded by previous layers.
      embed_edges=True,  # Embed raw features of the multi-mesh edges.
      edge_latent_size=dict(mesh=self.latent_size),
      node_latent_size=dict(mesh_nodes=self.latent_size),
      mlp_hidden_size=self.latent_size,
      mlp_num_hidden_layers=self.num_mlp_hidden_layers,
      num_message_passing_steps=self.num_message_passing_steps,
      use_layer_norm=True,
      conditional_normalization=self.conditional_normalization,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=False,
      # NOTE: segment_mean because number of edges is not balanced
      aggregate_edges_for_nodes_fn='segment_mean',
      name='mesh_gnn',
    )

    # Define step 1 of the decoder
    self._mesh2grid_gnn = DeepTypedGraphNet(
      embed_nodes=False,  # Node features already embdded by previous layers.
      embed_edges=True,  # Embed raw features of the mesh2grid edges.
      edge_latent_size=dict(mesh2grid=self.latent_size),
      node_latent_size=dict(mesh_nodes=self.latent_size, grid_nodes=self.latent_size),
      mlp_hidden_size=self.latent_size,
      mlp_num_hidden_layers=self.num_mlp_hidden_layers,
      num_message_passing_steps=1,
      use_layer_norm=True,
      conditional_normalization=self.conditional_normalization,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=False,
      # NOTE: segment_mean because number of edges is not balanced
      aggregate_edges_for_nodes_fn='segment_mean',
      name='mesh2grid_gnn',
    )

    # Define step 2 of the decoder
    self._grid2grid_gnn = DeepTypedGraphNet(
      # Require a specific node dimensionaly for the grid node outputs.
      node_output_size=dict(grid_nodes=self.num_outputs),
      embed_nodes=False,  # Node features already embdded by previous layers.
      embed_edges=True,  # Embed raw features of the grid2grid edges.
      edge_latent_size=dict(grid2grid=self.latent_size),
      node_latent_size=dict(grid_nodes=(self.latent_size * 2)),
      mlp_hidden_size=self.latent_size,
      mlp_num_hidden_layers=self.num_mlp_hidden_layers,
      num_message_passing_steps=self.num_message_passing_steps_grid,
      use_layer_norm=True,
      conditional_normalization=self.conditional_normalization,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=False,
      aggregate_edges_for_nodes_fn='segment_mean',
      name='grid2grid_gnn',
    )

  # TODO: Support 1D
  def _get_connections_by_mesh_node(self,
    idx_mesh_node_flat: int, overlap_factor: float = 1.0,
    ord_distance: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    # ord_distance = 1, 2, np.inf
    # overlap_factor == 1.0 :: there is no overlap

    # Get index of mesh node
    idx_mesh_node = (
      idx_mesh_node_flat // self.num_mesh_nodes[1],
      idx_mesh_node_flat % self.num_mesh_nodes[1]
    )

    # Set minimum radius based on the order of the distance
    r_min = np.linalg.norm(self.dz_mesh, ord=ord_distance) / 2

    # Get relative positions
    zeta_grid_rel = self.zeta_grid - self.zeta_mesh[idx_mesh_node[0], idx_mesh_node[1]]
    # Mirror relative positions because of periodic boudnary conditions
    zeta_grid_rel = np.where(zeta_grid_rel > 1., (zeta_grid_rel - 2.), zeta_grid_rel)

    # Compute distance
    # NOTE: Order of the norm determines
    dist_grid = np.linalg.norm(zeta_grid_rel, ord=ord_distance, axis=-1)

    # Get indices
    idx_grid_nodes = np.stack(np.where(dist_grid <= (overlap_factor * r_min)), axis=-1)
    idx_mesh_nodes = np.tile(np.array(idx_mesh_node), reps=(idx_grid_nodes.shape[0], 1))

    # Get flat indices
    idx_grid_nodes_flat = (idx_grid_nodes[:, 0] * self.num_grid_nodes[1] + idx_grid_nodes[:, 1])
    idx_mesh_nodes_flat = (idx_mesh_nodes[:, 0] * self.num_mesh_nodes[1] + idx_mesh_nodes[:, 1])

    return idx_grid_nodes_flat, idx_mesh_nodes_flat

  @staticmethod
  def _init_structural_features(
    zeta_sen: np.ndarray, zeta_rec: np.ndarray, idx_sen: list[int], idx_rec: list[int], node_freqs: int,
  ) -> Tuple[EdgeSet, NodeSet, NodeSet]:
    # NOTE: All inputs must be flattened: [num_nodes, num_dims]

    # Get number of nodes and the edges
    num_sen = zeta_sen.shape[0]
    num_rec = zeta_rec.shape[0]
    assert len(idx_sen) == len(idx_rec)
    num_edg = len(idx_sen)

    # Process coordinates
    phi_sen = np.pi * (zeta_sen + 1)  # [0, 2pi]
    phi_rec = np.pi * (zeta_rec + 1)  # [0, 2pi]

    # Define node features
    sender_node_feats = np.concatenate([
        np.concatenate([np.sin((k+1) * phi_sen), np.cos((k+1) * phi_sen)], axis=-1)
        for k in range(node_freqs)
      ], axis=-1)
    receiver_node_feats = np.concatenate([
        np.concatenate([np.sin((k+1) * phi_rec), np.cos((k+1) * phi_rec)], axis=-1)
        for k in range(node_freqs)
      ], axis=-1)

    # Build node sets
    sender_node_set = NodeSet(
      n_node=jnp.array([num_sen]),
      features=jnp.array(sender_node_feats)
    )
    receiver_node_set = NodeSet(
      n_node=jnp.array([num_rec]),
      features=jnp.array(receiver_node_feats)
    )

    # Define edge features
    z_ij = np.stack([
      (zeta_sen[s] - zeta_rec[r])
      for s, r in zip(idx_sen, idx_rec)
    ], axis=0)
    z_ij = np.where(z_ij < -1, z_ij + 2, z_ij)
    z_ij = np.where(z_ij >= 1, z_ij - 2, z_ij)
    d_ij = np.linalg.norm(z_ij, axis=-1, keepdims=True)
    edge_feats = np.concatenate([z_ij, d_ij], axis=-1)
    # Normalize edge features
    edge_feats = edge_feats / np.max(np.abs(edge_feats), axis=0, keepdims=True)

    # Build edge set
    edge_set = EdgeSet(
      n_edge=jnp.array([num_edg]),
      indices=EdgesIndices(
        senders=jnp.array(idx_sen),
        receivers=jnp.array(idx_rec)
      ),
      features=jnp.array(edge_feats),
    )

    return edge_set, sender_node_set, receiver_node_set

  def _init_grid2mesh_graph(self) -> TypedGraph:
    """Build Grid2Mesh graph."""

    edge_set, grid_node_set, mesh_node_set = self._init_structural_features(
      zeta_sen=self.zeta_grid.reshape(self._num_grid_nodes_tot, -1),
      zeta_rec=self.zeta_mesh.reshape(self._num_mesh_nodes_tot, -1),
      idx_sen=self.idx_grid2mesh_grid_nodes,
      idx_rec=self.idx_grid2mesh_mesh_nodes,
      node_freqs=self.node_coordinate_freqs,
    )

    graph = TypedGraph(
      context=Context(n_graph=jnp.array([1]), features=()),
      nodes={'grid_nodes': grid_node_set, 'mesh_nodes': mesh_node_set},
      edges={EdgeSetKey('grid2mesh', ('grid_nodes', 'mesh_nodes')): edge_set},
    )

    return graph

  def _init_mesh_graph(self) -> TypedGraph:
    """Build Mesh graph."""

    edge_set, mesh_node_set, _ = self._init_structural_features(
      zeta_sen=self.zeta_mesh.reshape(self._num_mesh_nodes_tot, -1),
      zeta_rec=self.zeta_mesh.reshape(self._num_mesh_nodes_tot, -1),
      idx_sen=self.idx_multimesh_send,
      idx_rec=self.idx_multimesh_recv,
      node_freqs=self.node_coordinate_freqs,
    )

    graph = TypedGraph(
      context=Context(n_graph=jnp.array([1]), features=()),
      nodes={'mesh_nodes': mesh_node_set},
      edges={EdgeSetKey('mesh', ('mesh_nodes', 'mesh_nodes')): edge_set},
    )

    return graph

  def _init_mesh2grid_graph(self) -> TypedGraph:
    """Build Mesh2Grid graph."""

    edge_set, mesh_node_set, grid_node_set = self._init_structural_features(
      zeta_sen=self.zeta_mesh.reshape(self._num_mesh_nodes_tot, -1),
      zeta_rec=self.zeta_grid.reshape(self._num_grid_nodes_tot, -1),
      idx_sen=self.idx_grid2mesh_mesh_nodes,
      idx_rec=self.idx_grid2mesh_grid_nodes,
      node_freqs=self.node_coordinate_freqs,
    )

    graph = TypedGraph(
      context=Context(n_graph=jnp.array([1]), features=()),
      nodes={'grid_nodes': grid_node_set, 'mesh_nodes': mesh_node_set},
      edges={EdgeSetKey('mesh2grid', ('mesh_nodes', 'grid_nodes')): edge_set},
    )

    return graph

  def _init_grid2grid_graph(self) -> TypedGraph:
    """Build Grid2Grid graph."""

    edge_set, grid_node_set, _ = self._init_structural_features(
      zeta_sen=self.zeta_grid.reshape(self._num_grid_nodes_tot, -1),
      zeta_rec=self.zeta_grid.reshape(self._num_grid_nodes_tot, -1),
      idx_sen=self.idx_grid2grid_send,
      idx_rec=self.idx_grid2grid_recv,
      node_freqs=self.node_coordinate_freqs,
    )

    graph = TypedGraph(
      context=Context(n_graph=jnp.array([1]), features=()),
      nodes={'grid_nodes': grid_node_set},
      edges={EdgeSetKey('grid2grid', ('grid_nodes', 'grid_nodes')): edge_set},
    )

    return graph

  def __call__(self,
    u_inp: Array,
    t_inp: Array = None,
    tau: Union[float, int] = None,
    specs: Array = None,
    key: flax.typing.PRNGKey = None,
  ) -> Array:
    """
    Inputs must be of shape (batch_size, 1, num_grid_nodes_0, num_grid_nodes_1, num_inputs)
    """

    assert u_inp.ndim == 3 + len(self.num_grid_nodes)
    batch_size = u_inp.shape[0]
    assert u_inp.shape[1] == 1
    assert u_inp.shape[2] == self.num_grid_nodes[0]
    assert u_inp.shape[3] == self.num_grid_nodes[1]
    assert u_inp.shape[-1] == self.num_outputs

    if specs is not None:
      assert specs.ndim == 2  # [batch_size, num_params]
      assert specs.shape[0] == batch_size

    if self.use_tau:
      assert tau is not None
      tau = jnp.array(tau, dtype=jnp.float32)
      if tau.size == 1:
        tau = jnp.tile(tau.reshape(1, 1), reps=(batch_size, 1))
    if self.use_t:
      assert t_inp is not None
      t_inp = jnp.array(t_inp, dtype=jnp.float32)
      if t_inp.size == 1:
        t_inp = jnp.tile(t_inp.reshape(1, 1), reps=(batch_size, 1))

    # Calculate, normalize, and concatenate derivatives
    if self.deriv_degree:
      d_inp = compute_derivatives(traj=u_inp, degree=self.deriv_degree)
      d_inp = normalize(
        arr=d_inp,
        shift=0.,
        scale=jnp.max(jnp.abs(d_inp), axis=(2, 3), keepdims=True),
      )
      u_inp = jnp.concatenate([u_inp, d_inp], axis=-1)

    # Prepare the grid node features
    # u -> [num_grid_nodes, batch_size, num_inputs]
    grid_node_features = jnp.moveaxis(
      u_inp, source=(0, 1, 2, 3),
      destination=(2, 3, 0, 1)
    ).reshape(self._num_grid_nodes_tot, batch_size, -1)
    # Concatente with forced features
    grid_node_features_forced = []
    if self.use_tau:
      grid_node_features_forced.append(
        jnp.tile(tau, reps=(self._num_grid_nodes_tot, 1, 1)))
    if self.use_t:
      grid_node_features_forced.append(
        jnp.tile(t_inp, reps=(self._num_grid_nodes_tot, 1, 1)))
    if specs is not None:
      grid_node_features_forced.append(
        jnp.repeat(specs[None, :, :], repeats=self._num_grid_nodes_tot, axis=0),)
    grid_node_features = jnp.concatenate(
      [grid_node_features, *grid_node_features_forced], axis=-1)

    # Transfer data for the grid to the mesh
    # -> [num_mesh_nodes, batch_size, latent_size], [num_grid_nodes, batch_size, latent_size]
    subkey, key = jax.random.split(key) if (key is not None) else (None, None)
    (latent_mesh_nodes, latent_grid_nodes) = self._run_grid2mesh_gnn(grid_node_features, tau, key=subkey)

    # Run message-passing in the multimesh.
    # -> [num_mesh_nodes, batch_size, latent_size]
    # TRY: Add edge dropout
    updated_latent_mesh_nodes = self._run_mesh_gnn(latent_mesh_nodes, tau)

    # Transfer data from the mesh to the grid.
    # -> [num_grid_nodes, batch_size, latent_size]
    subkey, key = jax.random.split(key) if (key is not None) else (None, None)
    updated_latent_grid_nodes = self._run_mesh2grid_gnn(updated_latent_mesh_nodes, latent_grid_nodes, tau, key=subkey)

    # Run message passing in the grid.
    # -> [num_grid_nodes, batch_size, num_outputs]
    output_grid_nodes = self._run_grid2grid_gnn(updated_latent_grid_nodes, latent_grid_nodes, tau)

    # Reshape the output to [batch_size, 1, num_grid_nodes, num_outputs]
    output = jnp.moveaxis(
      output_grid_nodes.reshape(
        self.num_grid_nodes[0], self.num_grid_nodes[1], batch_size, 1, self.num_outputs
      ),
      source=(0, 1, 2, 3),
      destination=(2, 3, 0, 1),
    )

    return output

  def _run_grid2mesh_gnn(self,
    grid_node_features: jnp.ndarray,
    tau: float,
    key: flax.typing.PRNGKey = None,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
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
          grid_nodes.features.astype(grid_node_features.dtype), bsz)
      ], axis=-1)
    )
    # To make sure capacity of the embedded is identical for the grid nodes and
    # the mesh nodes, we also append some dummy zero input features for the
    # mesh nodes.
    dummy_mesh_node_features = jnp.zeros(
        (self._num_mesh_nodes_tot,) + grid_node_features.shape[1:],
        dtype=grid_node_features.dtype)
    new_mesh_nodes = mesh_nodes._replace(
      features=jnp.concatenate([
        dummy_mesh_node_features,
        _add_batch_second_axis(
          mesh_nodes.features.astype(dummy_mesh_node_features.dtype), bsz)
      ], axis=-1)
    )

    # Get edges
    grid2mesh_edges_key = grid2mesh_graph.edge_key_by_name('grid2mesh')
    edges = grid2mesh_graph.edges[grid2mesh_edges_key]
    # Drop out edges randomly with the given probability
    if key is not None:
      n_edges_after = int((1 - self.p_dropout_edges_grid2mesh) * edges.features.shape[0])
      [new_edge_features, new_edge_senders, new_edge_receivers] = shuffle_arrays(
        key=key, arrays=[edges.features, edges.indices.senders, edges.indices.receivers])
      new_edge_features = new_edge_features[:n_edges_after]
      new_edge_senders = new_edge_senders[:n_edges_after]
      new_edge_receivers = new_edge_receivers[:n_edges_after]
    else:
      n_edges_after = edges.features.shape[0]
      new_edge_features = edges.features
      new_edge_senders = edges.indices.senders
      new_edge_receivers = edges.indices.receivers
    # Change edge feature dtype
    new_edge_features = new_edge_features.astype(dummy_mesh_node_features.dtype)
    # Broadcast edge structural features to the required batch size
    new_edge_features = _add_batch_second_axis(new_edge_features, bsz)
    # Build new edge set
    new_edges = EdgeSet(
      n_edge=jnp.array([n_edges_after]),
      indices=EdgesIndices(
        senders=new_edge_senders,
        receivers=new_edge_receivers,
      ),
      features=new_edge_features,
    )

    input_graph = grid2mesh_graph._replace(
      edges={grid2mesh_edges_key: new_edges},
      nodes={
        'grid_nodes': new_grid_nodes,
        'mesh_nodes': new_mesh_nodes
      })

    # Run the GNN.
    grid2mesh_out = self._grid2mesh_gnn(input_graph, condition=tau)
    latent_mesh_nodes = grid2mesh_out.nodes['mesh_nodes'].features
    latent_grid_nodes = grid2mesh_out.nodes['grid_nodes'].features

    return latent_mesh_nodes, latent_grid_nodes

  def _run_mesh_gnn(self,
    latent_mesh_nodes: Array,
    tau: float
  ) -> Array:
    """Runs the mesh_gnn, extracting updated latent mesh nodes."""

    bsz = latent_mesh_nodes.shape[1]
    mesh_graph = self._mesh_graph

    # Replace the node features
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
    output_graph = self._mesh_gnn(input_graph, condition=tau)
    output_mesh_nodes = output_graph.nodes['mesh_nodes'].features

    return output_mesh_nodes

  def _run_mesh2grid_gnn(self,
    updated_latent_mesh_nodes: Array,
    latent_grid_nodes: Array,
    tau: float,
    key: flax.typing.PRNGKey = None,
  ) -> Array:
    """Runs the mesh2grid_gnn, extracting the output grid nodes."""

    bsz = updated_latent_mesh_nodes.shape[1]
    mesh2grid_graph = self._mesh2grid_graph

    # NOTE: We don't need to add the structural node features, because these are
    # already part of the latent state, via the original Grid2Mesh gnn.
    mesh_nodes = mesh2grid_graph.nodes['mesh_nodes']
    grid_nodes = mesh2grid_graph.nodes['grid_nodes']
    new_mesh_nodes = mesh_nodes._replace(features=updated_latent_mesh_nodes)
    new_grid_nodes = grid_nodes._replace(features=latent_grid_nodes)

    # Get edges
    mesh2grid_edges_key = mesh2grid_graph.edge_key_by_name('mesh2grid')
    edges = mesh2grid_graph.edges[mesh2grid_edges_key]
    # Drop out edges randomly with the given probability
    if key is not None:
      n_edges_after = int((1 - self.p_dropout_edges_mesh2grid) * edges.features.shape[0])
      [new_edge_features, new_edge_senders, new_edge_receivers] = shuffle_arrays(
        key=key, arrays=[edges.features, edges.indices.senders, edges.indices.receivers])
      new_edge_features = new_edge_features[:n_edges_after]
      new_edge_senders = new_edge_senders[:n_edges_after]
      new_edge_receivers = new_edge_receivers[:n_edges_after]
    else:
      n_edges_after = edges.features.shape[0]
      new_edge_features = edges.features
      new_edge_senders = edges.indices.senders
      new_edge_receivers = edges.indices.receivers
    # Change edge feature dtype
    new_edge_features = new_edge_features.astype(latent_grid_nodes.dtype)
    # Broadcast edge structural features to the required batch size
    new_edge_features = _add_batch_second_axis(new_edge_features, bsz)
    # Build new edge set
    new_edges = EdgeSet(
      n_edge=jnp.array([n_edges_after]),
      indices=EdgesIndices(
        senders=new_edge_senders,
        receivers=new_edge_receivers,
      ),
      features=new_edge_features,
    )

    # Build the new graph
    input_graph = mesh2grid_graph._replace(
      edges={mesh2grid_edges_key: new_edges},
      nodes={
        'mesh_nodes': new_mesh_nodes,
        'grid_nodes': new_grid_nodes
      })

    # Run the GNN
    output_graph = self._mesh2grid_gnn(input_graph, condition=tau)
    output_grid_nodes = output_graph.nodes['grid_nodes'].features

    return output_grid_nodes

  def _run_grid2grid_gnn(self,
    latent_grid_nodes: Array,
    initial_latent_grid_nodes: Array,
    tau: float,
  ) -> Array:
    """Runs the grid2grid_gnn, extracting updated latent grid nodes."""

    bsz = latent_grid_nodes.shape[1]
    grid2grid_graph = self._grid2grid_graph

    # Replace the node features
    # NOTE: We don't need to add the structural node features, because these are
    # already part of the latent state, via the original Grid2Mesh gnn.
    concatenated_latent_grid_nodes = jnp.concatenate(
      [latent_grid_nodes, initial_latent_grid_nodes], axis=-1)
    nodes = grid2grid_graph.nodes['grid_nodes']
    nodes = nodes._replace(features=concatenated_latent_grid_nodes)

    # Add the structural edge features of this graph.
    # NOTE: We need the structural edge features, because it is the first
    # time we are seeing this particular set of edges.
    grid_edges_key = grid2grid_graph.edge_key_by_name('grid2grid')
    edges = grid2grid_graph.edges[grid_edges_key]
    new_edges = edges._replace(
      features=_add_batch_second_axis(
        edges.features.astype(latent_grid_nodes.dtype), bsz)
    )

    # Build the graph
    input_graph = grid2grid_graph._replace(
      edges={grid_edges_key: new_edges}, nodes={'grid_nodes': nodes}
    )

    # Run the GNN
    output_graph = self._grid2grid_gnn(input_graph, condition=tau)
    output_grid_nodes = output_graph.nodes['grid_nodes'].features

    return output_grid_nodes

def _add_batch_second_axis(data, bsz):
  # data [leading_dim, trailing_dim]
  assert data.ndim == 2
  ones = jnp.ones([bsz, 1], dtype=data.dtype)
  return data[:, None] * ones  # [leading_dim, batch, trailing_dim]
