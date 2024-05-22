import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors


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
        vmax = None
        vmin = None

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

def plot_trajectory(traj, idx_time, idx_traj=0, symmetric=True, ylabels=None):

  _HEIGHT_PER_ROW = 1.5
  _HEIGHT_MARGIN = .2
  _WIDTH_PER_COL = 1.5
  _WIDTH_MARGIN = .2

  n_vars = traj[0].shape[-1]
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
      vmax = None
      vmin = None

    for icol in range(len(idx_time)):
      h = axs[ivar, icol].imshow(
        traj[idx_traj, idx_time[icol], ..., ivar],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
      )
      if ivar == 0:
        axs[ivar, icol].set(title=f'timestep={idx_time[icol]}')

    plt.colorbar(h, ax=axs[ivar, :], fraction=.02)

  for ivar in range(n_vars):
    label = ylabels[ivar] if ylabels else f'Variable {ivar:02d}'
    axs[ivar, 0].set(ylabel=label);

  return fig, axs

def plot_estimations(u_gtr, u_prd, idx_time=-1, idx_traj=0, symmetric=True):

  _HEIGHT_PER_ROW = 2.5
  _HEIGHT_MARGIN = .2

  n_vars = u_gtr.shape[-1]
  if isinstance(symmetric, bool):
    symmetric = [symmetric] * n_vars

  fig, axs = plt.subplots(
    nrows=n_vars, ncols=3,
    figsize=(10, _HEIGHT_PER_ROW*n_vars+_HEIGHT_MARGIN),
    sharex=True, sharey=True,
  )
  axs = axs.reshape(n_vars, 3)

  u_err = (u_gtr - u_prd)

  for ivar in range(n_vars):
    vmax_gtr = np.max(np.abs(u_gtr[idx_traj, idx_time, ..., ivar]))

    h = axs[ivar, 0].imshow(
      u_gtr[idx_traj, idx_time, ..., ivar],
      cmap=(CMAP_BWR if symmetric[ivar] else CMAP_WRB),
      vmax=(vmax_gtr if symmetric[ivar] else None),
      vmin=(-vmax_gtr if symmetric[ivar] else None),
    )
    h = axs[ivar, 1].imshow(
      u_prd[idx_traj, idx_time, ..., ivar],
      cmap=(CMAP_BWR if symmetric[ivar] else CMAP_WRB),
      vmax=(vmax_gtr if symmetric[ivar] else None),
      vmin=(-vmax_gtr if symmetric[ivar] else None),
    )
    plt.colorbar(h, ax=axs[ivar, :2], fraction=.05)
    h = axs[ivar, 2].imshow(
      np.abs(u_err[idx_traj, idx_time, ..., ivar]),
      cmap=CMAP_WRB,
      vmin=0,
      vmax=np.max(np.abs(u_err[idx_traj, idx_time, ..., ivar])),
    )
    plt.colorbar(h, ax=axs[ivar, 2], fraction=.05)

  axs[0, 0].set(title='Ground-truth');
  axs[0, 1].set(title='Estimate');
  axs[0, 2].set(title='Absolute error');

  for ivar in range(n_vars):
    axs[ivar, 0].set(ylabel=f'Variable {ivar:02d}');

  return fig, axs

def animate_estimations(u_gtr, u_prd, u_err, idx_traj=0):
  _HEIGHT_PER_ROW = 1.5
  _HEIGHT_MARGIN = .2
  _WIDTH_PER_COL = 1.5
  _WIDTH_MARGIN = .2

  n_vars = u_gtr.shape[-1]
  n_time = u_gtr.shape[1]
  fig, axs = plt.subplots(
    nrows=n_vars, ncols=3,
    figsize=(_WIDTH_PER_COL*n_vars+_WIDTH_MARGIN, _HEIGHT_PER_ROW*3+_HEIGHT_MARGIN),
    sharex=True, sharey=True,
  )

  handlers_gtr = []
  handlers_err = []
  handlers_prd = []
  for ivar in range(n_vars):
    h = axs[ivar, 0].imshow(
      u_gtr[idx_traj, 0, ..., ivar],
      cmap=CMAP_BWR,
      vmin=-np.max(u_gtr[idx_traj, :, ..., ivar]),
      vmax=np.max(u_gtr[idx_traj, :, ..., ivar]),
    )
    handlers_gtr.append(h)
    h = axs[ivar, 1].imshow(
      u_prd[idx_traj, 0, ..., ivar],
      cmap=CMAP_BWR,
      vmin=-np.max(u_gtr[idx_traj, :, ..., ivar]),
      vmax=np.max(u_gtr[idx_traj, :, ..., ivar]),
    )
    handlers_prd.append(h)
    plt.colorbar(h, ax=axs[ivar, :2], fraction=.1)
    h = axs[ivar, 2].imshow(
      np.abs(u_err[idx_traj, 0, ..., ivar]),
      cmap=CMAP_WRB,
      vmin=0,
      vmax=np.max(np.abs(u_err[idx_traj, :, ..., ivar])),
    )
    handlers_err.append(h)
    plt.colorbar(h, ax=axs[ivar, 2], fraction=.2)

  axs[0, 0].set(title='Ground-truth');
  axs[0, 1].set(title='Estimate');
  axs[0, 2].set(title='Absolute error');

  for ivar in range(n_vars):
    axs[ivar, 0].set(ylabel=f'Variable {ivar:02d}');

  def update(frame):
    fig.suptitle(f'timestep = {frame}')
    for ivar in range(n_vars):
      handlers_gtr[ivar].set_data(u_gtr[idx_traj, frame, ..., ivar])
      handlers_prd[ivar].set_data(u_prd[idx_traj, frame, ..., ivar])
      handlers_err[ivar].set_data(u_err[idx_traj, frame, ..., ivar])

  ani = animation.FuncAnimation(fig=fig, func=update, frames=n_time, interval=150)

  return ani, (fig, axs)

def plot_error_accumulation(u_err, idx_time, idx_traj=0):
  _HEIGHT_PER_ROW = 1.5
  _HEIGHT_MARGIN = .2
  _WIDTH_PER_COL = 1.5
  _WIDTH_MARGIN = .2

  # Because the initial state is removed from the errors
  idx_time=[i-1 for i in idx_time]

  n_vars = u_err.shape[-1]
  fig, axs = plt.subplots(
    nrows=n_vars, ncols=len(idx_time),
    figsize=(_WIDTH_PER_COL*len(idx_time)+_WIDTH_MARGIN, _HEIGHT_PER_ROW*n_vars+_HEIGHT_MARGIN),
    sharex=True, sharey=True,
  )

  for ivar in range(n_vars):
    for icol in range(len(idx_time)):
      h = axs[ivar, icol].imshow(
        np.abs(u_err[idx_traj, idx_time[icol], ..., ivar]),
        cmap=CMAP_WRB,
        vmin=0,
        vmax=np.max(u_err[idx_traj, idx_time, ..., :]),
      )
      if ivar == 0:
        axs[ivar, icol].set(title=f'timestep={idx_time[icol]+1}')
  plt.colorbar(h, ax=axs, fraction=.02)

  for ivar in range(n_vars):
    axs[ivar, 0].set(ylabel=f'Variable {ivar:02d}');

  return fig, axs

def plot_ensemble(u_gtr, u_prd, idx_time=-1, idx_traj=0, symmetric=True):

  _HEIGHT_PER_ROW = 2.5
  _HEIGHT_MARGIN = .2

  n_vars = u_prd.shape[-1]
  if isinstance(symmetric, bool):
    symmetric = [symmetric] * n_vars

  fig, axs = plt.subplots(
    nrows=n_vars, ncols=3,
    figsize=(10, _HEIGHT_PER_ROW*n_vars+_HEIGHT_MARGIN),
    sharex=True, sharey=True,
  )

  u_prd_mean = np.mean(u_prd, axis=0)
  u_prd_std = np.std(u_prd, axis=0)
  u_err = (u_gtr - u_prd_mean)

  for ivar in range(n_vars):
    vmax = np.max(np.abs(u_prd_mean[idx_traj, idx_time, ..., ivar]))

    h = axs[ivar, 0].imshow(
      u_prd_mean[idx_traj, idx_time, ..., ivar],
      cmap=(CMAP_BWR if symmetric[ivar] else CMAP_WRB),
      vmax=(vmax if symmetric[ivar] else None),
      vmin=(-vmax if symmetric[ivar] else None),
    )
    plt.colorbar(h, ax=axs[ivar, 0], fraction=.05)
    h = axs[ivar, 1].imshow(
      u_prd_std[idx_traj, idx_time, ..., ivar],
      cmap=CMAP_WRB,
      vmin=0,
      vmax=None,
    )
    plt.colorbar(h, ax=axs[ivar, 1], fraction=.05)
    h = axs[ivar, 2].imshow(
      np.abs(u_err[idx_traj, idx_time, ..., ivar]),
      cmap=CMAP_WRB,
      vmin=0,
      vmax=None,
    )
    plt.colorbar(h, ax=axs[ivar, 2], fraction=.05)

  axs[0, 0].set(title='Ensemble mean');
  axs[0, 1].set(title='Ensemble std');
  axs[0, 2].set(title='Absolute error');

  for ivar in range(n_vars):
    axs[ivar, 0].set(ylabel=f'Variable {ivar:02d}');

  return fig, axs