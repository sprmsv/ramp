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
    c_inp: Array = None,
    x_inp: Array = None,
    x_out: Array = None,
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

  x: Array
  num_outputs: int
  periodic: bool
  rmesh_levels: int = 1
  subsample_factor: float = 2
  overlap_factor_p2r: int = 2.0
  overlap_factor_r2p: int = 2.0
  mlp_hidden_layers: int = 1
  node_latent_size: int = 128
  edge_latent_size: int = 128
  mlp_hidden_size: int = 128
  processor_steps: int = 18
  concatenate_t: bool = True
  concatenate_tau: bool = True
  conditional_normalization: bool = True
  conditional_norm_latent_size: int = 16
  node_coordinate_freqs: int = 4
  p_dropout_edges_p2r: int = 0.5
  p_dropout_edges_r2r: int = 0.5
  p_dropout_edges_r2p: int = 0.5

  def _check_coordinates(self, x: Array) -> None:
    assert x is not None
    assert x.ndim == 2
    assert x.shape[1] <= 3
    assert x.min() >= -1
    assert x.max() <= +1

  def _check_function(self, u: Array, x: Array) -> None:
    assert u is not None
    assert u.ndim == 4
    assert u.shape[1] == 1
    assert u.shape[2] == x.shape[0]

  def _get_supported_points(self,
    centers: Array,
    points: Array,
    radii: Array,
    ord_distance: int = 2,
  ) -> Array:
    """ord_distance can be 1, 2, or np.inf"""

    assert jnp.all(radii < 1.0)

    # Get relative coordinates
    rel = points[:, None] - centers
    # Mirror relative positions because of periodic boudnary conditions
    if self.periodic:
      rel = jnp.where(rel >= 1., (rel - 2.), rel)
      rel = jnp.where(rel < -1., (rel + 2.), rel)

    # Compute distance
    # NOTE: Order of the norm determines the shape of the sub-regions
    distance = jnp.linalg.norm(rel, ord=ord_distance, axis=-1)

    # Get indices
    # -> [idx_point, idx_center]
    idx_nodes = jnp.stack(jnp.where(distance <= radii), axis=-1)

    return idx_nodes

  def _init_structural_features(self,
    x_sen: Array, x_rec: Array,
    idx_sen: list[int], idx_rec: list[int],
    node_freqs: int, max_edge_length: float,
    domain_sen: list[int] = None,
    domain_rec: list[int] = None,
    shifts: list[Array] = None,
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
      # MODIFY: Unify the mirroring with the below method in r2r
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

  def _init_graphs(self, key, x_inp: Array, x_out: Array) -> dict[str, TypedGraph]:

    # Randomly sub-sample pmesh to get rmesh
    if key is None:
      key = jax.random.PRNGKey(0)
    x_rmesh = _subsample_pointset(key=key, x=x_inp, factor=self.subsample_factor)

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
      _boundary_linspace = jnp.linspace(-1, 1, 64, endpoint=True).reshape(-1, 1)
      x_boundary = jnp.concatenate([
        jnp.concatenate([-jnp.ones_like(_boundary_linspace), _boundary_linspace], axis=1),
        jnp.concatenate([_boundary_linspace, +jnp.ones_like(_boundary_linspace)], axis=1),
        jnp.concatenate([+jnp.ones_like(_boundary_linspace), _boundary_linspace], axis=1),
        jnp.concatenate([_boundary_linspace, -jnp.ones_like(_boundary_linspace)], axis=1),
      ])
      x_rmesh = jnp.concatenate([x_rmesh, x_boundary])

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

    # Compute minimum support radius of each rmesh node
    minimum_radius = _compute_minimum_support_radii(x_rmesh)

    def _init_p2r_graph() -> TypedGraph:
      """Constructrs the encoder graph (pmesh to rmesh)"""

      # Set the sub-region radii
      radius = self.overlap_factor_p2r * minimum_radius

      # Get indices of supported points
      idx_nodes = self._get_supported_points(
        centers=x_rmesh,
        points=x_inp,
        radii=radius,
      )

      # Get the initial features
      edge_set, pmesh_node_set, rmesh_node_set = self._init_structural_features(
        x_sen=x_inp,
        x_rec=x_rmesh,
        idx_sen=idx_nodes[:, 0],
        idx_rec=idx_nodes[:, 1],
        node_freqs=self.node_coordinate_freqs,
        max_edge_length=np.max(radius),
      )

      # Construct the graph
      graph = TypedGraph(
        context=Context(n_graph=jnp.array([1]), features=()),
        nodes={'pnodes': pmesh_node_set, 'rnodes': rmesh_node_set},
        edges={EdgeSetKey('p2r', ('pnodes', 'rnodes')): edge_set},
      )

      return graph

    def _init_r2r_graph() -> TypedGraph:
      """Constructrs the processor graph (rmesh to rmesh)"""

      # Define edges and their corresponding -extended- domain
      edges = []
      domains = []
      for level in range(self.rmesh_levels):
        # Sub-sample the rmesh
        _rmesh_size = int(x_rmesh.shape[0] / (self.subsample_factor ** level))
        _x_rmesh = x_rmesh[:_rmesh_size]
        if self.periodic:
          # Repeat the rmesh in periodic directions
          _x_rmesh_extended = jnp.concatenate(
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
        max_edge_length=(2. * jnp.sqrt(x_inp.shape[1])),
        shifts=jnp.array(_domain_shifts).squeeze(1),
        domain_sen=[i for (i, j) in domains],
        domain_rec=[j for (i, j) in domains],
      )

      # Construct the graph
      graph = TypedGraph(
        context=Context(n_graph=jnp.array([1]), features=()),
        nodes={'rnodes': rmesh_node_set},
        edges={EdgeSetKey('r2r', ('rnodes', 'rnodes')): edge_set},
      )

      return graph

    def _init_r2p_graph() -> TypedGraph:
      """Constructrs the decoder graph (rmesh to pmesh)"""

      # Set the sub-region radii
      radius = self.overlap_factor_r2p * minimum_radius

      # Get indices of supported points
      idx_nodes = self._get_supported_points(
        centers=x_rmesh,
        points=x_out,
        radii=radius,
      )

      # Get the initial features
      edge_set, rmesh_node_set, pmesh_node_set = self._init_structural_features(
        x_sen=x_rmesh,
        x_rec=x_out,
        idx_sen=idx_nodes[:, 1],
        idx_rec=idx_nodes[:, 0],
        node_freqs=self.node_coordinate_freqs,
        max_edge_length=np.max(radius),
      )

      # Construct the graph
      graph = TypedGraph(
        context=Context(n_graph=jnp.array([1]), features=()),
        nodes={'pnodes': pmesh_node_set, 'rnodes': rmesh_node_set},
        edges={EdgeSetKey('r2p', ('rnodes', 'pnodes')): edge_set},
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
      embed_nodes=True,  # Embed raw features of the physical and the regional meshes
      embed_edges=True,  # Embed raw features of the p2r edges.
      edge_latent_size=dict(p2r=self.edge_latent_size),
      node_latent_size=dict(rnodes=self.node_latent_size, pnodes=self.node_latent_size),
      mlp_hidden_size=self.mlp_hidden_size,
      mlp_num_hidden_layers=self.mlp_hidden_layers,
      num_message_passing_steps=1,
      use_layer_norm=True,
      conditional_normalization=self.conditional_normalization,
      conditional_norm_latent_size=self.conditional_norm_latent_size,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=True,
      aggregate_edges_for_nodes_fn='segment_mean',
      aggregate_normalization=None,
      name='p2r_gnn',
    )

    # Define the processor
    self._r2r_gnn = DeepTypedGraphNet(
      embed_nodes=False,  # Node features already embdded by previous layers.
      embed_edges=True,  # Embed raw features of the multi-mesh edges.
      edge_latent_size=dict(r2r=self.edge_latent_size),
      node_latent_size=dict(rnodes=self.node_latent_size),
      mlp_hidden_size=self.mlp_hidden_size,
      mlp_num_hidden_layers=self.mlp_hidden_layers,
      num_message_passing_steps=self.processor_steps,
      use_layer_norm=True,
      conditional_normalization=self.conditional_normalization,
      conditional_norm_latent_size=self.conditional_norm_latent_size,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=False,
      # NOTE: segment_mean because number of edges is not balanced
      aggregate_edges_for_nodes_fn='segment_mean',
      name='r2r_gnn',
    )

    # Define step 1 of the decoder
    self._r2p_gnn = DeepTypedGraphNet(
      # NOTE: with variable mesh, the output pnode features must be embedded first
      # NOTE: without variable mesh, there is no need for embeddings
      embed_nodes=self.variable_mesh,
      embed_edges=True,  # Embed raw features of the r2p edges.
      edge_latent_size=dict(r2p=self.edge_latent_size),
      node_latent_size=dict(rnodes=self.node_latent_size, pnodes=self.node_latent_size),
      mlp_hidden_size=self.mlp_hidden_size,
      mlp_num_hidden_layers=self.mlp_hidden_layers,
      num_message_passing_steps=1,
      use_layer_norm=True,
      conditional_normalization=self.conditional_normalization,
      conditional_norm_latent_size=self.conditional_norm_latent_size,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=False,
      # NOTE: segment_mean because number of edges is not balanced
      aggregate_edges_for_nodes_fn='segment_mean',
      name='r2p_gnn',
    )

  def setup(self):
    # NOTE: There are a few architectural considerations for variable mesh
    # NOTE: variable_mesh=True means that the input and the output mesh can be different
    # NOTE: Check usages of this attribute
    self.variable_mesh = False

    if self.x is not None:
      self._check_coordinates(x=self.x)
      self.graphs = self._init_graphs(key=None, x_inp=self.x, x_out=self.x)
    else:
      self.graphs = None

    self._init_gnns()

  @staticmethod
  def _reorder_features(feats: Array, num_nodes: int) -> Array:
    batch_size = feats.shape[1]
    num_feats = feats.shape[-1]
    feats = feats.reshape(num_nodes, batch_size, 1, num_feats)
    output = jnp.moveaxis(feats, source=(0, 1, 2), destination=(2, 0, 1))
    return output

  def _run_gnns(self,
    graphs: dict[str, TypedGraph],
    pnode_features: Array,
    tau: float,
    key: flax.typing.PRNGKey = None,
  ) -> Array:

    def _run_p2r_gnn(
      p2r_graph: TypedGraph,
      pnode_features: Array,
      tau: float,
      key: flax.typing.PRNGKey = None,
    ) -> tuple[Array, Array]:
      """Runs the p2r GNN, extracting latent physical and regional nodes."""

      # Get batch size
      batch_size = pnode_features.shape[1]

      # Concatenate node structural features with input features
      pnodes = p2r_graph.nodes['pnodes']
      rnodes = p2r_graph.nodes['rnodes']
      new_pnodes = pnodes._replace(
        features=jnp.concatenate([
          pnode_features,
          _add_batch_second_axis(
            pnodes.features.astype(pnode_features.dtype), batch_size)
        ], axis=-1)
      )
      # To make sure capacity of the embedded is identical for the physical nodes and
      # the regional nodes, we also append some dummy zero input features for the
      # regional nodes.
      dummy_rnode_features = jnp.zeros(
          (rnodes.n_node.item(),) + pnode_features.shape[1:],
          dtype=pnode_features.dtype)
      new_rnodes = rnodes._replace(
        features=jnp.concatenate([
          dummy_rnode_features,
          _add_batch_second_axis(
            rnodes.features.astype(dummy_rnode_features.dtype), batch_size)
        ], axis=-1)
      )

      # Get edges
      p2r_edges_key = p2r_graph.edge_key_by_name('p2r')
      edges = p2r_graph.edges[p2r_edges_key]
      # Drop out edges randomly with the given probability
      if key is not None:
        n_edges_after = int((1 - self.p_dropout_edges_p2r) * edges.features.shape[0])
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
      new_edge_features = new_edge_features.astype(dummy_rnode_features.dtype)
      # Broadcast edge structural features to the required batch size
      new_edge_features = _add_batch_second_axis(new_edge_features, batch_size)
      # Build new edge set
      new_edges = EdgeSet(
        n_edge=jnp.array([n_edges_after]),
        indices=EdgesIndices(
          senders=new_edge_senders,
          receivers=new_edge_receivers,
        ),
        features=new_edge_features,
      )

      input_graph = p2r_graph._replace(
        edges={p2r_edges_key: new_edges},
        nodes={
          'pnodes': new_pnodes,
          'rnodes': new_rnodes
        })

      # Run the GNN.
      p2r_out = self._p2r_gnn(input_graph, condition=tau)
      latent_rnodes = p2r_out.nodes['rnodes'].features
      latent_pnodes = p2r_out.nodes['pnodes'].features

      return latent_rnodes, latent_pnodes

    def _run_r2r_gnn(
      r2r_graph: TypedGraph,
      latent_rnodes: Array,
      tau: float,
      key: flax.typing.PRNGKey = None,
    ) -> Array:
      """Runs the r2r GNN, extracting updated latent regional nodes."""

      # Get batch size
      batch_size = latent_rnodes.shape[1]

      # Replace the node features
      # NOTE: We don't need to add the structural node features, because these are
      # already part of  the latent state, via the original p2r gnn.
      rnodes = r2r_graph.nodes['rnodes']
      new_rnodes = rnodes._replace(features=latent_rnodes)

      # Get edges
      r2r_edges_key = r2r_graph.edge_key_by_name('r2r')
      # NOTE: We are assuming here that the r2r gnn uses a single set of edge keys
      # named 'r2r' for the edges and that it uses a single set of nodes named 'rnodes'
      msg = ('The setup currently requires to only have one kind of edge in the mesh GNN.')
      assert len(r2r_graph.edges) == 1, msg
      edges = r2r_graph.edges[r2r_edges_key]
      # Drop out edges randomly with the given probability
      # NOTE: We need the structural edge features, because it is the first
      # time we are seeing this particular set of edges.
      if key is not None:
        n_edges_after = int((1 - self.p_dropout_edges_r2r) * edges.features.shape[0])
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
      new_edge_features = new_edge_features.astype(latent_rnodes.dtype)
      # Broadcast edge structural features to the required batch size
      new_edge_features = _add_batch_second_axis(new_edge_features, batch_size)
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
      input_graph = r2r_graph._replace(
        edges={r2r_edges_key: new_edges},
        nodes={'rnodes': new_rnodes},
      )

      # Run the GNN
      output_graph = self._r2r_gnn(input_graph, condition=tau)
      output_mesh_nodes = output_graph.nodes['rnodes'].features

      return output_mesh_nodes

    def _run_r2p_gnn(
      r2p_graph: TypedGraph,
      updated_latent_rnodes: Array,
      latent_pnodes: Array,
      tau: float,
      key: flax.typing.PRNGKey = None,
    ) -> Array:
      """Runs the r2p GNN, extracting the output physical nodes."""

      # Get batch size
      batch_size = updated_latent_rnodes.shape[1]

      # NOTE: We don't need to add the structural node features, because these are
      # already part of the latent state, via the original p2r gnn.
      rnodes = r2p_graph.nodes['rnodes']
      pnodes = r2p_graph.nodes['pnodes']
      new_rnodes = rnodes._replace(features=updated_latent_rnodes)
      if self.variable_mesh:
        # NOTE: We can't use latent pnodes of the input mesh for the output mesh
        # TRY: Make sure that this does not harm the performance with fixed mesh
        # If it works, change the architecture, flowcharts, etc.
        new_pnodes = pnodes._replace(features=_add_batch_second_axis(pnodes.features, batch_size))
      else:
        new_pnodes = pnodes._replace(features=latent_pnodes)

      # Get edges
      r2p_edges_key = r2p_graph.edge_key_by_name('r2p')
      edges = r2p_graph.edges[r2p_edges_key]
      # Drop out edges randomly with the given probability
      if key is not None:
        n_edges_after = int((1 - self.p_dropout_edges_r2p) * edges.features.shape[0])
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
      new_edge_features = new_edge_features.astype(latent_pnodes.dtype)
      # Broadcast edge structural features to the required batch size
      new_edge_features = _add_batch_second_axis(new_edge_features, batch_size)
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
      input_graph = r2p_graph._replace(
        edges={r2p_edges_key: new_edges},
        nodes={
          'rnodes': new_rnodes,
          'pnodes': new_pnodes
        })

      # Run the GNN
      output_graph = self._r2p_gnn(input_graph, condition=tau)
      output_pnodes = output_graph.nodes['pnodes'].features

      return output_pnodes

    # Transfer data for the physical mesh to the regional mesh
    # -> [num_nodes, batch_size, latent_size]
    subkey, key = jax.random.split(key) if (key is not None) else (None, None)
    (latent_rnodes, latent_pnodes) = _run_p2r_gnn(graphs['p2r'], pnode_features, tau, key=subkey)
    self.sow(
      col='intermediates', name='pnodes_encoded',
      value=self._reorder_features(latent_pnodes, graphs['p2r'].nodes['pnodes'].n_node.item())
    )
    self.sow(
      col='intermediates', name='rnodes_encoded',
      value=self._reorder_features(latent_rnodes, graphs['p2r'].nodes['rnodes'].n_node.item())
    )

    # Run message-passing in the regional mesh
    # -> [num_rnodes, batch_size, latent_size]
    subkey, key = jax.random.split(key) if (key is not None) else (None, None)
    updated_latent_rnodes = _run_r2r_gnn(graphs['r2r'], latent_rnodes, tau, key=subkey)
    self.sow(
      col='intermediates', name='rnodes_processed',
      value=self._reorder_features(updated_latent_rnodes, graphs['r2r'].nodes['rnodes'].n_node.item())
    )

    # Transfer data from the regional mesh to the physical mesh
    # -> [num_pnodes_out, batch_size, latent_size]
    subkey, key = jax.random.split(key) if (key is not None) else (None, None)
    output_pnodes = _run_r2p_gnn(graphs['r2p'], updated_latent_rnodes, latent_pnodes, tau, key=subkey)
    self.sow(
      col='intermediates', name='pnodes_decoded',
      value=self._reorder_features(output_pnodes, graphs['r2p'].nodes['pnodes'].n_node.item())
    )

    return output_pnodes

  def __call__(self,
    u_inp: Array,
    c_inp: Array = None,
    x_inp: Array = None,
    x_out: Array = None,
    t_inp: Array = None,  # TMP: Support None
    tau: Union[float, int] = None,  # TMP: Support None
    key: flax.typing.PRNGKey = None,
  ) -> Array:
    """
    Inputs must be of shape (batch_size, 1, num_physical_nodes, num_inputs)
    """

    # Check input and output coordinates
    if self.x is not None:
      assert x_inp is None
      assert x_out is None
      x_inp = self.x
      x_out = self.x
      graphs = self.graphs
    else:
      self._check_coordinates(x_inp)
      self._check_coordinates(x_out)
      subkey, key = jax.random.split(key) if (key is not None) else (None, None)
      graphs = self._init_graphs(key=subkey, x_inp=x_inp, x_out=x_out)

    # Check input functions
    self._check_function(u_inp, x=x_inp)
    if c_inp is not None:
      self._check_function(c_inp, x=x_inp)
    assert u_inp.shape[3] == self.num_outputs

    # Read dimensions
    batch_size = u_inp.shape[0]
    num_pnodes_inp = x_inp.shape[0]
    num_pnodes_out = x_out.shape[0]

    # Prepare the time channel
    if self.concatenate_t:
      assert t_inp is not None
      t_inp = jnp.array(t_inp, dtype=jnp.float32)
      if t_inp.size == 1:
        t_inp = jnp.tile(t_inp.reshape(1, 1), reps=(batch_size, 1))
    # Prepare the time difference channel
    if self.concatenate_tau:
      assert tau is not None
      tau = jnp.array(tau, dtype=jnp.float32)
      if tau.size == 1:
        tau = jnp.tile(tau.reshape(1, 1), reps=(batch_size, 1))

    # Concatenate the known coefficients to the channels of the input function
    if c_inp is not None:
      u_inp = jnp.concatenate([u_inp, c_inp], axis=-1)

    # Prepare the physical node features
    # u -> [num_pnodes_inp, batch_size, num_inputs]
    pnode_features = jnp.moveaxis(
      u_inp, source=(0, 1, 2, 3),
      destination=(1, 3, 0, 2)
    ).squeeze(axis=3)

    # Concatente with forced features
    pnode_features_forced = []
    if self.concatenate_tau:
      pnode_features_forced.append(
        jnp.tile(tau, reps=(num_pnodes_inp, 1, 1)))
    if self.concatenate_t:
      pnode_features_forced.append(
        jnp.tile(t_inp, reps=(num_pnodes_inp, 1, 1)))
    pnode_features = jnp.concatenate(
      [pnode_features, *pnode_features_forced], axis=-1)

    # Run the GNNs
    subkey, key = jax.random.split(key) if (key is not None) else (None, None)
    output_pnodes = self._run_gnns(graphs=graphs, pnode_features=pnode_features, tau=tau, key=subkey)

    # Reshape the output to [batch_size, 1, num_pnodes_out, num_outputs]
    # [num_pnodes_out, batch_size, num_outputs] -> u
    output = self._reorder_features(output_pnodes, num_pnodes_out)

    return output

def _subsample_pointset(key, x: Array, factor: float) -> Array:
  x = jnp.array(x)
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
