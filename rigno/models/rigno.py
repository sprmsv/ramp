from typing import Tuple, Union

import flax.typing
import jax.numpy as jnp
import jax.random
import numpy as np
from flax import linen as nn
from scipy.spatial import Delaunay

from rigno.graph.typed_graph import (
    TypedGraph, EdgeSet, EdgeSetKey,
    EdgesIndices, NodeSet, Context)
from rigno.models.graphnet import DeepTypedGraphNet
from rigno.utils import Array, shuffle_arrays


class AbstractOperator(nn.Module):
  def setup(self):
    raise NotImplementedError

  def __call__(self,
    u_inp: Array,
    t_inp: Array = None,
    tau: Union[float, int] = None,
    key: flax.typing.PRNGKey = None,
  ) -> Array:
    raise NotImplementedError

  @property
  def configs(self):
    configs = {
      attr: self.__getattr__(attr)
      for attr in self.__annotations__.keys() if attr != 'parent'
    }
    return configs

class RIGNO(AbstractOperator):
  """TODO: Add docstrings"""
  # NOTE: Only fixed dx is supported for now

  num_outputs: int
  periodic: bool = True
  concatenate_t: bool = True
  concatenate_tau: bool = True
  conditional_normalization: bool = False
  conditional_norm_latent_size: int = 16
  node_latent_size: int = 128
  edge_latent_size: int = 128
  num_mlp_hidden_layers: int = 2
  mlp_hidden_size: int = 128
  num_message_passing_steps: int = 18
  num_message_passing_steps_grid: int = 2
  node_coordinate_freqs: int = 1
  p_dropout_edges_grid2mesh: int = 0.
  p_dropout_edges_multimesh: int = 0.
  p_dropout_edges_mesh2grid: int = 0.

  # TMP
  fixed_mesh: bool = False
  x_pmesh: Array = None
  rmesh_subsample_factor: float = 2
  rmesh_levels: int = 1
  overlap_factor_p2r: int = 1.0
  overlap_factor_r2p: int = 1.0

  def _check_coordinates(self, x: Array) -> None:
    assert x is not None
    assert x.ndim == 2
    assert x.shape[1] <= 3
    assert x.min() >= -1
    assert x.max() <= +1

  def _get_supported_points(self,
    centers: np.ndarray,
    points: np.ndarray,
    radii: np.ndarray,
    ord_distance: int = 2,
  ) -> np.ndarray:
    """ord_distance can be 1, 2, or np.inf"""

    assert all(radii[1] < 1.0)

    # Get relative coordinates
    rel = points[:, None] - centers
    # Mirror relative positions because of periodic boudnary conditions
    if self.periodic:
      rel = np.where(rel >= 1., (rel - 2.), rel)
      rel = np.where(rel < -1., (rel + 2.), rel)

    # Compute distance
    # NOTE: Order of the norm determines the shape of the sub-regions
    distance = np.linalg.norm(rel, ord=ord_distance, axis=-1)

    # Get indices
    # -> [idx_point, idx_center]
    idx_nodes = np.stack(np.where(distance <= radii), axis=-1)

    return idx_nodes

  def _init_structural_features(self,
    x_sen: np.ndarray, x_rec: np.ndarray,
    idx_sen: list[int], idx_rec: list[int],
    node_freqs: int, max_edge_length: float,
    domain_sen: list[int] = None,
    domain_rec: list[int] = None,
    shifts: list[np.ndarray] = None,
  ) -> Tuple[EdgeSet, NodeSet, NodeSet]:

    # Get number of nodes and the edges
    num_sen = x_sen.shape[0]
    num_rec = x_rec.shape[0]
    assert len(idx_sen) == len(idx_rec)
    num_edg = len(idx_sen)

    # Process coordinates
    phi_sen = np.pi * (x_sen + 1)  # [0, 2pi]
    phi_rec = np.pi * (x_rec + 1)  # [0, 2pi]

    # Define node features
    # NOTE: Sinusoidal features don't need normalization
    if self.periodic:
      sender_node_feats = np.concatenate([
          np.concatenate([np.sin((k+1) * phi_sen), np.cos((k+1) * phi_sen)], axis=-1)
          for k in range(node_freqs)
        ], axis=-1)
      receiver_node_feats = np.concatenate([
          np.concatenate([np.sin((k+1) * phi_rec), np.cos((k+1) * phi_rec)], axis=-1)
          for k in range(node_freqs)
        ], axis=-1)
    else:
      sender_node_feats = np.concatenate([x_sen], axis=-1)
      receiver_node_feats = np.concatenate([x_rec], axis=-1)

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
      (x_sen[s] - x_rec[r])
      for s, r in zip(idx_sen, idx_rec)
    ], axis=0)
    assert np.all(np.abs(z_ij) <= 2.)
    if self.periodic:
      # NOTE: For p2r and r2p, mirror the large relative coordinates
      # TODO: Unify the mirroring with the below method in r2r
      if shifts is None:
        z_ij = np.where(z_ij < -1.0, z_ij + 2, z_ij)
        z_ij = np.where(z_ij >= 1.0, z_ij - 2, z_ij)
      # NOTE: For the r2r multi-mesh, use extended domain indices and shifts
      else:
        z_ij = np.stack([
          ((x_sen[s] + shifts[domain_sen[s]]) - (x_rec[r] + shifts[domain_rec[r]]))
          for s, r in zip(idx_sen, idx_rec)
        ], axis=0)
    d_ij = np.linalg.norm(z_ij, axis=-1, keepdims=True)
    # Normalize and concatenate edge features
    assert np.all(np.abs(z_ij) <= max_edge_length)
    assert np.all(np.abs(d_ij) <= max_edge_length)
    z_ij = z_ij / max_edge_length
    d_ij = d_ij / max_edge_length
    edge_feats = np.concatenate([z_ij, d_ij], axis=-1)

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

  def _init_graphs(self, key, x_pmesh: Array):
    # TMP TODO: re-use the function in __call__

    # Randomly sub-sample pmesh to get rmesh
    if key is None:
      key = jax.random.PRNGKey(0)
    x_rmesh = self._subsample_pointset(key=key, x=x_pmesh, factor=self.rmesh_subsample_factor)

    # Domain shifts for periodic BC
    _domain_shifts = [
      np.array([[0., 0.]]),  # C
      np.array([[-2, 0.]]),  # W
      np.array([[-2, +2]]),  # NW
      np.array([[0., +2]]),  # N
      np.array([[+2, +2]]),  # NE
      np.array([[+2, 0.]]),  # E
      np.array([[+2, -2]]),  # SE
      np.array([[0., -2]]),  # S
      np.array([[-2, -2]]),  # SW
    ]

    # NOTE: Always include boundary nodes for non-periodic BC
    # TODO: Update based on boundary node settings  # TMP
    if not self.periodic:
      _boundary_linspace = np.linspace(-1, 1, 64, endpoint=True).reshape(-1, 1)
      x_boundary = np.concatenate([
        np.concatenate([-np.ones_like(_boundary_linspace), _boundary_linspace], axis=1),
        np.concatenate([_boundary_linspace, +np.ones_like(_boundary_linspace)], axis=1),
        np.concatenate([+np.ones_like(_boundary_linspace), _boundary_linspace], axis=1),
        np.concatenate([_boundary_linspace, -np.ones_like(_boundary_linspace)], axis=1),
      ])
      x_rmesh = np.concatenate([x_rmesh, x_boundary])

    def _compute_minimum_support_radii(x: Array):
      if self.periodic:
        x_extended = np.concatenate(
          [x + _domain_shifts[idx] for idx in range(len(_domain_shifts))], axis=0)
        tri = Delaunay(points=x_extended)
      else:
        tri = Delaunay(points=x)

      medians = _compute_triangulation_medians(tri)
      radii = np.zeros(shape=(x.shape[0],))
      for s, simplex in enumerate(tri.simplices):
        for v in range(simplex.shape[0]):
          if simplex[v] < x.shape[0]:
            m = medians[s, v]
            radii[simplex[v]] = max(m, radii[simplex[v]])

      return radii

    def _init_p2r_graph() -> TypedGraph:
      """Constructrs the encoder graph (pmesh to rmesh)"""

      # Set the sub-region radii
      radius = self.overlap_factor_p2r * _compute_minimum_support_radii(x_rmesh)

      # Get indices of supported points
      idx_nodes = self._get_supported_points(
        center=x_rmesh,
        points=x_pmesh,
        radii=radius,
      )

      # Get the initial features
      edge_set, pmesh_node_set, rmesh_node_set = self._init_structural_features(
        x_sen=x_pmesh,
        x_rec=x_rmesh,
        idx_sen=idx_nodes[:, 0],
        idx_rec=idx_nodes[:, 1],
        node_freqs=self.node_coordinate_freqs,
        max_edge_length=np.max(radius),
      )

      # Construct the graph
      graph = TypedGraph(
        context=Context(n_graph=jnp.array([1]), features=()),
        nodes={'pmesh_nodes': pmesh_node_set, 'rmesh_nodes': rmesh_node_set},
        edges={EdgeSetKey('p2r', ('pmesh_nodes', 'rmesh_nodes')): edge_set},
      )

      return graph

    def _init_r2r_graph() -> TypedGraph:
      """Constructrs the processor graph (rmesh to rmesh)"""

      # Define edges and their corresponding -extended- domain
      edges = []
      domains = []
      for level in range(self.rmesh_levels):
        # Sub-sample the rmesh
        _rmesh_size = int(x_rmesh.shape[0] / (self.rmesh_subsample_factor ** level))
        _x_rmesh = x_rmesh[:_rmesh_size]
        if self.periodic:
          # Repeat the rmesh in periodic directions
          _x_rmesh_extended = np.concatenate(
            [_x_rmesh + _domain_shifts[idx] for idx in range(len(_domain_shifts))], axis=0)
          tri = Delaunay(points=_x_rmesh_extended)
        else:
          tri = Delaunay(points=_x_rmesh)
        # Construct a triangulation and get the edges
        _extended_edges = _get_edges_from_triangulation(tri)
        # Keep the relevant edges
        for edge in _extended_edges:
          domain = tuple([i // _rmesh_size for i in edge])
          edge = tuple([i % _rmesh_size for i in edge])
          if (domain == (0, 0)) or (self.periodic and (0 in domain)):
            if edge not in edges:
              domains.append(domain)
              edges.append(edge)

      # Set the initial features
      edge_set, rmesh_node_set, _ = self._init_structural_features(
        x_sen=x_rmesh,
        x_rec=x_rmesh,
        idx_sen=[i for (i, j) in edges],
        idx_rec=[j for (i, j) in edges],
        node_freqs=self.node_coordinate_freqs,
        max_edge_length=(2. * np.sqrt(x_pmesh.shape[1])),
        shifts=_domain_shifts,
        domain_sen=[i for (i, j) in domains],
        domain_rec=[j for (i, j) in domains],
      )

      # Construct the graph
      graph = TypedGraph(
        context=Context(n_graph=jnp.array([1]), features=()),
        nodes={'rmesh_nodes': rmesh_node_set},
        edges={EdgeSetKey('mesh', ('rmesh_nodes', 'rmesh_nodes')): edge_set},
      )

      return graph

    def _init_r2p_graph() -> TypedGraph:
      """Constructrs the decoder graph (rmesh to pmesh)"""

      # Set the sub-region radii
      radius = self.overlap_factor_r2p * _compute_minimum_support_radii(x_rmesh)

      # Get indices of supported points
      idx_nodes = self._get_supported_points(
        center=x_rmesh,
        points=x_pmesh,
        radii=radius,
      )

      # Get the initial features
      edge_set, rmesh_node_set, pmesh_node_set = self._init_structural_features(
        x_sen=x_rmesh,
        x_rec=x_pmesh,
        idx_sen=idx_nodes[:, 1],
        idx_rec=idx_nodes[:, 0],
        node_freqs=self.node_coordinate_freqs,
        max_edge_length=np.max(radius),
      )

      # Construct the graph
      graph = TypedGraph(
        context=Context(n_graph=jnp.array([1]), features=()),
        nodes={'pmesh_nodes': pmesh_node_set, 'rmesh_nodes': rmesh_node_set},
        edges={EdgeSetKey('r2p', ('rmesh_nodes', 'pmesh_nodes')): edge_set},
      )

      return graph

    graphs = {
      'p2r': _init_p2r_graph(),
      'r2r': _init_r2r_graph(),
      'r2p': _init_r2p_graph(),
    }

    return graphs

  def _init_gnns(self):

    # Define the encoder
    self._p2r_gnn = DeepTypedGraphNet(
      embed_nodes=True,  # Embed raw features of the grid and mesh nodes.
      embed_edges=True,  # Embed raw features of the grid2mesh edges.
      edge_latent_size=dict(grid2mesh=self.edge_latent_size),
      node_latent_size=dict(mesh_nodes=self.node_latent_size, grid_nodes=self.node_latent_size),
      mlp_hidden_size=self.mlp_hidden_size,
      mlp_num_hidden_layers=self.num_mlp_hidden_layers,
      num_message_passing_steps=1,
      use_layer_norm=True,
      conditional_normalization=self.conditional_normalization,
      conditional_norm_latent_size=self.conditional_norm_latent_size,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=True,
      aggregate_edges_for_nodes_fn='segment_mean',
      aggregate_normalization=None,
      name='grid2mesh_gnn',
    )

    # Define the processor
    self._r2r_gnn = DeepTypedGraphNet(
      embed_nodes=False,  # Node features already embdded by previous layers.
      embed_edges=True,  # Embed raw features of the multi-mesh edges.
      edge_latent_size=dict(mesh=self.edge_latent_size),
      node_latent_size=dict(mesh_nodes=self.node_latent_size),
      mlp_hidden_size=self.mlp_hidden_size,
      mlp_num_hidden_layers=self.num_mlp_hidden_layers,
      num_message_passing_steps=self.num_message_passing_steps,
      use_layer_norm=True,
      conditional_normalization=self.conditional_normalization,
      conditional_norm_latent_size=self.conditional_norm_latent_size,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=False,
      # NOTE: segment_mean because number of edges is not balanced
      aggregate_edges_for_nodes_fn='segment_mean',
      name='mesh_gnn',
    )

    # Define step 1 of the decoder
    self._r2p_gnn = DeepTypedGraphNet(
      embed_nodes=False,  # Node features already embdded by previous layers.
      embed_edges=True,  # Embed raw features of the mesh2grid edges.
      edge_latent_size=dict(mesh2grid=self.edge_latent_size),
      node_latent_size=dict(mesh_nodes=self.node_latent_size, grid_nodes=self.node_latent_size),
      mlp_hidden_size=self.mlp_hidden_size,
      mlp_num_hidden_layers=self.num_mlp_hidden_layers,
      num_message_passing_steps=1,
      use_layer_norm=True,
      conditional_normalization=self.conditional_normalization,
      conditional_norm_latent_size=self.conditional_norm_latent_size,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=False,
      # NOTE: segment_mean because number of edges is not balanced
      aggregate_edges_for_nodes_fn='segment_mean',
      name='mesh2grid_gnn',
    )

  def setup(self):

    if self.fixed_mesh:
      self._check_coordinates(self.x_pmesh)
      self.graphs = self._init_graphs(key=None, x_pmesh=self.x_pmesh)
    else:
      self.graphs = None

    self._init_gnns()

  def features2grid(feats, num_nodes) -> Array:
    batch_size = feats.shape[1]
    num_feats = feats.shape[-1]
    output = jnp.moveaxis(
      feats.reshape(
        num_nodes[0], num_nodes[1], batch_size, 1, num_feats
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
    grid2mesh_graph = self._p2r_graph

    # Concatenate node structural features with input features
    grid_nodes = grid2mesh_graph.nodes['pmesh_nodes']
    mesh_nodes = grid2mesh_graph.nodes['rmesh_nodes']
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
    grid2mesh_edges_key = grid2mesh_graph.edge_key_by_name('p2r')
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
        'pmesh_nodes': new_grid_nodes,
        'rmesh_nodes': new_mesh_nodes
      })

    # Run the GNN.
    grid2mesh_out = self._p2r_gnn(input_graph, condition=tau)
    latent_mesh_nodes = grid2mesh_out.nodes['rmesh_nodes'].features
    latent_grid_nodes = grid2mesh_out.nodes['pmesh_nodes'].features

    return latent_mesh_nodes, latent_grid_nodes

  def _run_mesh_gnn(self,
    latent_mesh_nodes: Array,
    tau: float,
    key: flax.typing.PRNGKey = None,
  ) -> Array:
    """Runs the mesh_gnn, extracting updated latent mesh nodes."""

    bsz = latent_mesh_nodes.shape[1]
    mesh_graph = self._r2r_graph

    # Replace the node features
    # NOTE: We don't need to add the structural node features, because these are
    # already part of  the latent state, via the original Grid2Mesh gnn.
    mesh_nodes = mesh_graph.nodes['rmesh_nodes']
    new_mesh_nodes = mesh_nodes._replace(features=latent_mesh_nodes)

    # Get edges
    mesh_edges_key = mesh_graph.edge_key_by_name('mesh')
    # NOTE: We are assuming here that the mesh gnn uses a single set of edge keys
    # named 'mesh' for the edges and that it uses a single set of nodes named 'rmesh_nodes'
    msg = ('The setup currently requires to only have one kind of edge in the mesh GNN.')
    assert len(mesh_graph.edges) == 1, msg
    edges = mesh_graph.edges[mesh_edges_key]
    # Drop out edges randomly with the given probability
    # NOTE: We need the structural edge features, because it is the first
    # time we are seeing this particular set of edges.
    if key is not None:
      n_edges_after = int((1 - self.p_dropout_edges_multimesh) * edges.features.shape[0])
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
    new_edge_features = new_edge_features.astype(latent_mesh_nodes.dtype)
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

    # Build the graph
    input_graph = mesh_graph._replace(
      edges={mesh_edges_key: new_edges},
      nodes={'rmesh_nodes': new_mesh_nodes},
    )

    # Run the GNN
    output_graph = self._r2r_gnn(input_graph, condition=tau)
    output_mesh_nodes = output_graph.nodes['rmesh_nodes'].features

    return output_mesh_nodes

  def _run_mesh2grid_gnn(self,
    updated_latent_mesh_nodes: Array,
    latent_grid_nodes: Array,
    tau: float,
    key: flax.typing.PRNGKey = None,
  ) -> Array:
    """Runs the mesh2grid_gnn, extracting the output grid nodes."""

    bsz = updated_latent_mesh_nodes.shape[1]
    mesh2grid_graph = self._r2p_graph

    # NOTE: We don't need to add the structural node features, because these are
    # already part of the latent state, via the original Grid2Mesh gnn.
    mesh_nodes = mesh2grid_graph.nodes['rmesh_nodes']
    grid_nodes = mesh2grid_graph.nodes['pmesh_nodes']
    new_mesh_nodes = mesh_nodes._replace(features=updated_latent_mesh_nodes)
    new_grid_nodes = grid_nodes._replace(features=latent_grid_nodes)

    # Get edges
    mesh2grid_edges_key = mesh2grid_graph.edge_key_by_name('r2p')
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
        'rmesh_nodes': new_mesh_nodes,
        'pmesh_nodes': new_grid_nodes
      })

    # Run the GNN
    output_graph = self._r2p_gnn(input_graph, condition=tau)
    output_grid_nodes = output_graph.nodes['pmesh_nodes'].features

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
    nodes = grid2grid_graph.nodes['pmesh_nodes']
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
      edges={grid_edges_key: new_edges}, nodes={'pmesh_nodes': nodes}
    )

    # Run the GNN
    output_graph = self._grid2grid_gnn(input_graph, condition=tau)
    output_grid_nodes = output_graph.nodes['pmesh_nodes'].features

    return output_grid_nodes

  def __call__(self,
    u_inp: Array,
    c_inp: Array = None,
    x_inp: Array = None,
    x_out: Array = None,
    t_inp: Array = None,
    tau: Union[float, int] = None,
    key: flax.typing.PRNGKey = None,
  ) -> Array:
    """
    Inputs must be of shape (batch_size, 1, num_grid_nodes_0, num_grid_nodes_1, num_inputs)
    """

    if self.fixed_mesh:
      assert x_inp is None
      assert x_out is None
    else:
      self._check_coordinates(x_inp)
      self._check_coordinates(x_out)

    assert u_inp.ndim == 3 + len(self.num_grid_nodes)
    batch_size = u_inp.shape[0]
    assert u_inp.shape[1] == 1
    assert u_inp.shape[2] == self.num_grid_nodes[0]
    assert u_inp.shape[3] == self.num_grid_nodes[1]
    assert u_inp.shape[-1] == self.num_outputs

    if self.concatenate_tau:
      assert tau is not None
      tau = jnp.array(tau, dtype=jnp.float32)
      if tau.size == 1:
        tau = jnp.tile(tau.reshape(1, 1), reps=(batch_size, 1))
    if self.concatenate_t:
      assert t_inp is not None
      t_inp = jnp.array(t_inp, dtype=jnp.float32)
      if t_inp.size == 1:
        t_inp = jnp.tile(t_inp.reshape(1, 1), reps=(batch_size, 1))

    # Prepare the grid node features
    # u -> [num_grid_nodes, batch_size, num_inputs]
    grid_node_features = jnp.moveaxis(
      u_inp, source=(0, 1, 2, 3),
      destination=(2, 3, 0, 1)
    ).reshape(self._num_grid_nodes_tot, batch_size, -1)
    # Concatente with forced features
    grid_node_features_forced = []
    if self.concatenate_tau:
      grid_node_features_forced.append(
        jnp.tile(tau, reps=(self._num_grid_nodes_tot, 1, 1)))
    if self.concatenate_t:
      grid_node_features_forced.append(
        jnp.tile(t_inp, reps=(self._num_grid_nodes_tot, 1, 1)))
    grid_node_features = jnp.concatenate(
      [grid_node_features, *grid_node_features_forced], axis=-1)

    # Transfer data for the grid to the mesh
    # -> [num_mesh_nodes, batch_size, latent_size], [num_grid_nodes, batch_size, latent_size]
    subkey, key = jax.random.split(key) if (key is not None) else (None, None)
    (latent_mesh_nodes, latent_grid_nodes) = self._run_grid2mesh_gnn(grid_node_features, tau, key=subkey)
    self.sow(
      col='intermediates', name='grid_encoded',
      value=self.features2grid(latent_grid_nodes, self.num_grid_nodes)
    )
    self.sow(
      col='intermediates', name='mesh_encoded',
      value=self.features2grid(latent_mesh_nodes, self.num_mesh_nodes)
    )

    # Run message-passing in the multimesh.
    # -> [num_mesh_nodes, batch_size, latent_size]
    subkey, key = jax.random.split(key) if (key is not None) else (None, None)
    updated_latent_mesh_nodes = self._run_mesh_gnn(latent_mesh_nodes, tau, key=subkey)
    self.sow(
      col='intermediates', name='mesh_processed',
      value=self.features2grid(updated_latent_mesh_nodes, self.num_mesh_nodes)
    )

    # Transfer data from the mesh to the grid.
    # -> [num_grid_nodes, batch_size, latent_size]
    subkey, key = jax.random.split(key) if (key is not None) else (None, None)
    updated_latent_grid_nodes = self._run_mesh2grid_gnn(updated_latent_mesh_nodes, latent_grid_nodes, tau, key=subkey)
    self.sow(
      col='intermediates', name='grid_decoded',
      value=self.features2grid(updated_latent_grid_nodes, self.num_grid_nodes)
    )

    # Run message passing in the grid.
    # -> [num_grid_nodes, batch_size, num_outputs]
    output_grid_nodes = self._run_grid2grid_gnn(updated_latent_grid_nodes, latent_grid_nodes, tau)
    self.sow(
      col='intermediates', name='grid_output',
      value=self.features2grid(output_grid_nodes, self.num_grid_nodes)
    )

    # Reshape the output to [batch_size, 1, num_grid_nodes, num_outputs]
    output = self.features2grid(output_grid_nodes, self.num_grid_nodes)

    return output

def _subsample_pointset(key, x: Array, factor: float) -> Array:
  x_shuffled, = shuffle_arrays(key, [x])
  return x_shuffled[:int(x.shape[0] / factor)]

def _get_edges_from_triangulation(tri: Delaunay, bidirectional: bool = True):
  indptr, indices = tri.vertex_neighbor_vertices
  edges = [(k, l) for k in range(tri.points.shape[0]) for l in indices[indptr[k]:indptr[k+1]]]
  if bidirectional:
    edges += [(l, k) for (k, l) in edges]
  return edges

def _compute_triangulation_medians(tri: Delaunay) -> Array:
  # Only in 2D

  edges = np.zeros(shape=tri.simplices.shape)
  medians = np.zeros(shape=tri.simplices.shape)
  for i in range(tri.simplices.shape[1]):
    points = tri.points[np.delete(tri.simplices, i, axis=1)]
    points = [p.squeeze(1) for p in np.split(points, axis=1, indices_or_sections=2)]
    edges[:, i] = np.linalg.norm(np.subtract(*points), axis=1)
  for i in range(tri.simplices.shape[1]):
    medians[:, i] = .67 * np.sqrt((2 * np.sum(np.power(np.delete(edges, i, axis=1), 2), axis=1) - np.power(edges[:, i], 2)) / 4)

  return medians


def _add_batch_second_axis(data, bsz):
  """
  Adds a batch axis by repeating the input

  input: [leading_dim, trailing_dim]
  output: [leading_dim, batch, trailing_dim]
  """

  assert data.ndim == 2
  ones = jnp.ones([bsz, 1], dtype=data.dtype)
  return data[:, None] * ones
