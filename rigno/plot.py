import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
import numpy as np
import pandas as pd
import seaborn as sns


C_BLACK = '#000000'
C_WHITE = '#ffffff'
C_BLUE = '#093691'
C_RED = '#911b09'
C_BLACK_BLUEISH = '#011745'
C_BLACK_REDDISH = '#380801'
C_WHITE_BLUEISH = '#dce5f5'
C_WHITE_REDDISH = '#f5dcdc'

CMAP_BBR = matplotlib.colors.LinearSegmentedColormap.from_list(
  'blue_black_red',
  [C_WHITE_BLUEISH, C_BLUE, C_BLACK, C_RED, C_WHITE_REDDISH],
  N=200,
)
CMAP_BWR = matplotlib.colors.LinearSegmentedColormap.from_list(
  'blue_white_red',
  [C_BLACK_BLUEISH, C_BLUE, C_WHITE, C_RED, C_BLACK_REDDISH],
  N=200,
)
CMAP_WRB = matplotlib.colors.LinearSegmentedColormap.from_list(
  'white_red_black',
  [C_WHITE, C_RED, C_BLACK],
  N=200,
)

SCATTER_SETTINGS = dict(marker='s', s=4, alpha=.9)

# TODO: Update
def animate(trajs, idx_traj=0, symmetric=True, cmaps=CMAP_BBR, vertical=True):

  if not isinstance(trajs, list):
    trajs = [trajs]
  n_trjs = len(trajs)

  if not isinstance(cmaps, list):
    cmaps = [cmaps] * n_trjs
  assert len(trajs) == len(cmaps)

  if not isinstance(symmetric, list):
    symmetric = [symmetric] * n_trjs
  assert len(trajs) == len(symmetric)

  n_vars = trajs[0].shape[-1]
  n_time = trajs[0].shape[1]

  if vertical:
    fig, axs = plt.subplots(
      nrows=n_vars, ncols=n_trjs,
      figsize=(5*n_trjs, 4*n_vars)
    )
  else:
    fig, axs = plt.subplots(
      nrows=n_trjs, ncols=n_vars,
      figsize=(5*n_vars, 4*n_trjs)
    )

  handlers = []
  for i in range(n_vars):
    for j in range(n_trjs):
      if symmetric[j]:
        vmax = np.max(np.abs(trajs[j][idx_traj, :, ..., i]))
        vmin = -vmax
      else:
        vmax = np.max(trajs[j][idx_traj, :, ..., i])
        vmin = np.min(trajs[j][idx_traj, :, ..., i])

      idx = (n_trjs * i + j) if vertical else (n_vars * j + i)
      h = axs.flatten()[idx].imshow(
        trajs[j][idx_traj, 0, ..., i],
        cmap=cmaps[j],
        vmin=vmin,
        vmax=vmax,
      )
      plt.colorbar(h)
      handlers.append(h)

  def update(frame):
    for i in range(n_vars):
      for j in range(n_trjs):
        idx = (n_trjs * i + j) if vertical else (n_vars * j + i)
        handlers[idx].set_data(trajs[j][idx_traj, frame, ..., i])

  ani = animation.FuncAnimation(fig=fig, func=update, frames=n_time, interval=150)

  return ani, (fig, axs)

def plot_grid_trajectory(traj, idx_time, idx_traj=0, symmetric=True, ylabels=None):

  _HEIGHT_PER_ROW = 1.5
  _HEIGHT_MARGIN = .2
  _WIDTH_PER_COL = 1.5
  _WIDTH_MARGIN = .2

  n_vars = traj.shape[-1]
  if isinstance(symmetric, bool):
    symmetric = [symmetric] * n_vars

  fig, axs = plt.subplots(
    nrows=n_vars, ncols=len(idx_time),
    figsize=(_WIDTH_PER_COL*len(idx_time)+_WIDTH_MARGIN, _HEIGHT_PER_ROW*n_vars+_HEIGHT_MARGIN),
    sharex=True, sharey=True,
  )
  if (n_vars == 1) and len(idx_time) == 1:
    axs = np.array(axs)
  axs = axs.reshape(n_vars, len(idx_time))

  for ivar in range(n_vars):
    if symmetric[ivar]:
      cmap = CMAP_BWR
      vmax = np.max(np.abs(traj[idx_traj, idx_time, ..., ivar]))
      vmin = -vmax
    else:
      cmap = CMAP_WRB
      vmax = np.max(traj[idx_traj, idx_time, ..., ivar])
      vmin = np.min(traj[idx_traj, idx_time, ..., ivar])

    for icol in range(len(idx_time)):
      h = axs[ivar, icol].imshow(
        traj[idx_traj, idx_time[icol], ..., ivar],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
      )
      if ivar == 0:
        axs[ivar, icol].set(title=f'$t=t_{{{idx_time[icol]}}}$')

    plt.colorbar(h, ax=axs[ivar, :], fraction=.02)

  for ivar in range(n_vars):
    label = ylabels[ivar] if ylabels else f'Variable {ivar:02d}'
    axs[ivar, 0].set(ylabel=label);

  for ax in axs.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

  return fig, axs

def plot_estimates(u_inp, u_gtr, u_prd, x_inp, x_out, symmetric=True, names=None):

  _HEIGHT_PER_ROW = 2.5
  _HEIGHT_MARGIN = .2

  n_vars = u_gtr.shape[-1]
  if isinstance(symmetric, bool):
    symmetric = [symmetric] * n_vars

  fig, axs = plt.subplots(
    nrows=n_vars, ncols=4,
    figsize=(12, _HEIGHT_PER_ROW*n_vars+_HEIGHT_MARGIN),
    sharex=True, sharey=True,
  )
  axs = axs.reshape(n_vars, 4)

  u_err = (u_gtr - u_prd)

  for ivar in range(n_vars):
    vmax_inp = np.max(u_inp[:, ivar])
    vmax_gtr = np.max(u_gtr[:, ivar])
    vmax_prd = np.max(u_prd[:, ivar])
    vmax = max(vmax_inp, vmax_gtr, vmax_prd)
    vmin_inp = np.min(u_inp[:, ivar])
    vmin_gtr = np.min(u_gtr[:, ivar])
    vmin_prd = np.min(u_prd[:, ivar])
    vmin = min(vmin_inp, vmin_gtr, vmin_prd)
    vmax_abs = max(np.abs(vmax), np.abs(vmin))

    vmin_gtr = np.min(np.abs(u_gtr[:, ivar]))

    h = axs[ivar, 0].scatter(
      x=x_inp[:, 0],
      y=x_inp[:, 1],
      c=u_inp[:, ivar],
      cmap=(CMAP_BWR if symmetric[ivar] else CMAP_WRB),
      vmax=(vmax_abs if symmetric[ivar] else vmax),
      vmin=(-vmax_abs if symmetric[ivar] else vmin),
      **SCATTER_SETTINGS,
    )
    h = axs[ivar, 1].scatter(
      x=x_out[:, 0],
      y=x_out[:, 1],
      c=u_gtr[:, ivar],
      cmap=(CMAP_BWR if symmetric[ivar] else CMAP_WRB),
      vmax=(vmax_abs if symmetric[ivar] else vmax),
      vmin=(-vmax_abs if symmetric[ivar] else vmin),
      **SCATTER_SETTINGS,
    )
    h = axs[ivar, 2].scatter(
      x=x_out[:, 0],
      y=x_out[:, 1],
      c=u_prd[:, ivar],
      cmap=(CMAP_BWR if symmetric[ivar] else CMAP_WRB),
      vmax=(vmax_abs if symmetric[ivar] else vmax),
      vmin=(-vmax_abs if symmetric[ivar] else vmin),
      **SCATTER_SETTINGS,
    )
    plt.colorbar(h, ax=axs[ivar, 0:3], fraction=.1)
    h = axs[ivar, 3].scatter(
      x=x_out[:, 0],
      y=x_out[:, 1],
      c=np.abs(u_err[:, ivar]),
      cmap=CMAP_WRB,
      vmin=0,
      vmax=np.max(np.abs(u_err[:, ivar])),
      **SCATTER_SETTINGS,
    )
    plt.colorbar(h, ax=axs[ivar, 3], fraction=.1)

  axs[0, 0].set(title='Input');
  axs[0, 1].set(title='Ground-truth');
  axs[0, 2].set(title='Model estimate');
  axs[0, 3].set(title='Absolute error');

  if n_vars > 1:
    for ivar in range(n_vars):
      label = names[ivar] if names else f'Variable {ivar:02d}'
      axs[ivar, 0].set(ylabel=label);

  for ax in axs.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set(xlim=[0, 1])
    ax.set(ylim=[0, 1])

  return fig, axs

def plot_ensemble(u_gtr, u_ens, x, symmetric=True, names=None):

  _HEIGHT_PER_ROW = 2.5
  _HEIGHT_MARGIN = .2

  n_vars = u_ens.shape[-1]
  if isinstance(symmetric, bool):
    symmetric = [symmetric] * n_vars

  fig, axs = plt.subplots(
    nrows=n_vars, ncols=3,
    figsize=(10, _HEIGHT_PER_ROW*n_vars+_HEIGHT_MARGIN),
    sharex=True, sharey=True,
  )
  axs = axs.reshape(n_vars, 3)

  u_ens_mean = np.mean(u_ens, axis=0)
  u_ens_std = np.std(u_ens, axis=0)
  u_err = (u_gtr - u_ens_mean)

  for ivar in range(n_vars):
    vmax = np.max(np.abs(u_gtr[:, ivar]))

    h = axs[ivar, 0].scatter(
      x=x[:, 0],
      y=x[:, 1],
      c=u_ens_mean[:, ivar],
      cmap=(CMAP_BWR if symmetric[ivar] else CMAP_WRB),
      vmax=(vmax if symmetric[ivar] else None),
      vmin=(-vmax if symmetric[ivar] else None),
      **SCATTER_SETTINGS,
    )
    plt.colorbar(h, ax=axs[ivar, 0], fraction=.1)
    h = axs[ivar, 1].scatter(
      x=x[:, 0],
      y=x[:, 1],
      c=u_ens_std[:, ivar],
      cmap=CMAP_WRB,
      vmin=0,
      vmax=None,
      **SCATTER_SETTINGS,
    )
    plt.colorbar(h, ax=axs[ivar, 1], fraction=.1)
    h = axs[ivar, 2].scatter(
      x=x[:, 0],
      y=x[:, 1],
      c=np.abs(u_err[:, ivar]),
      cmap=CMAP_WRB,
      vmin=0,
      vmax=None,
      **SCATTER_SETTINGS,
    )
    plt.colorbar(h, ax=axs[ivar, 2], fraction=.1)

  axs[0, 0].set(title='Ensemble mean');
  axs[0, 1].set(title='Ensemble std');
  axs[0, 2].set(title='Absolute error');

  if n_vars > 1:
    for ivar in range(n_vars):
      label = names[ivar] if names else f'Variable {ivar:02d}'
      axs[ivar, 0].set(ylabel=label);

  for ax in axs.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

  return fig, axs

def plot_error_vs_time(df: pd.DataFrame, idx_fn: int, variable_title: str = 'variable', palette: str = None) -> sns.FacetGrid:
  g = sns.FacetGrid(
    data=(df[(df['error'] > 0.)]),
    hue='variable',
    palette=palette,
    height=4,
    aspect=.8,
  );
  g.set_titles(col_template='{col_name}');
  g.map(sns.scatterplot, 't', 'error', marker='o', s=30, alpha=1);
  g.map(sns.lineplot, 't', 'error', linewidth=2, alpha=.8);
  g.add_legend(title=variable_title);
  g.set_ylabels(label='Error (%)');

  sns.move_legend(g, loc='right', bbox_to_anchor=(1.02, .5))

  for ax in g.axes.flatten():
    ax.axvline(x=idx_fn, linestyle='--', color='black', linewidth=2, alpha=.5);
    ax.set_xticks([t for t in df['t'].unique() if (t%2 == 0)], minor=False);
    ax.set_xticks([t for t in df['t'].unique()], minor=True);
    ax.grid(which='major');

  return g

# TODO: Update
def plot_intermediates(features, idx_traj: int = 0, idx_time: int = 0, share_cmap: bool = False):
  _HEIGHT_PER_ROW = 2
  _HEIGHT_MARGIN = .2
  _WIDTH_PER_COL = 2
  _WIDTH_MARGIN = .2

  COL_WRAP = 6
  n_vars = features.shape[-1]
  if n_vars > COL_WRAP:
    n_rows = n_vars // COL_WRAP
    n_cols = COL_WRAP
  else:
    n_rows = 1
    n_cols = n_vars

  fig, axs = plt.subplots(
    nrows=n_rows,
    ncols=n_cols,
    figsize=(_WIDTH_PER_COL*n_cols+_WIDTH_MARGIN, _HEIGHT_PER_ROW*n_rows+_HEIGHT_MARGIN),
    sharex=True, sharey=True,
  )
  if (n_cols * n_rows) == 1:
    axs = np.array(axs)
  axs = axs.reshape(n_rows * n_cols)

  for ivar in range(n_vars):
    cmap = CMAP_BWR
    if share_cmap:
      vmax = np.max(np.abs(features[idx_traj, idx_time, ..., :]))
    else:
      vmax = np.max(np.abs(features[idx_traj, idx_time, ..., ivar]))
    vmin = -vmax
    h = axs[ivar].imshow(
      features[idx_traj, idx_time, ..., ivar],
      cmap=cmap,
      vmin=vmin,
      vmax=vmax,
    )

  for ax in axs.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

  return fig, axs
