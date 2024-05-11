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
    fig, axs = plt.subplots(nrows=n_vars, ncols=n_trjs, figsize=(5*n_trjs, 4*n_vars))
  else:
    fig, axs = plt.subplots(nrows=n_trjs, ncols=n_vars, figsize=(5*n_vars, 4*n_trjs))

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

def plot_trajectory(traj, idx_time, idx_traj=0, symmetric=True, cmap=CMAP_BBR, ylabels=None):

  n_vars = traj[0].shape[-1]

  fig, axs = plt.subplots(nrows=n_vars, ncols=len(idx_time), figsize=(8, 1.5*n_vars+.2), sharex=True, sharey=True)
  if (n_vars == 1) and len(idx_time) == 1:
    axs = np.array(axs)
  axs = axs.reshape(n_vars, len(idx_time))

  for ivar in range(n_vars):
    if symmetric:
      vmax = np.max(np.abs(traj[idx_traj, idx_time, ..., ivar]))
      vmin = -vmax
    else:
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

def plot_estimations(u_gtr, u_prd, u_err, idx_time=-1, idx_traj=0):
  n_vars = u_gtr.shape[-1]
  fig, axs = plt.subplots(nrows=n_vars, ncols=3, figsize=(10, 2.5*n_vars), sharex=True, sharey=True)
  fig.tight_layout()

  for ivar in range(n_vars):
    h = axs[ivar, 0].imshow(
      u_gtr[idx_traj, idx_time, ..., ivar],
      cmap=CMAP_BWR,
      vmin=-np.max(u_gtr[idx_traj, idx_time, ..., ivar]),
      vmax=np.max(u_gtr[idx_traj, idx_time, ..., ivar]),
    )
    h = axs[ivar, 1].imshow(
      u_prd[idx_traj, idx_time, ..., ivar],
      cmap=CMAP_BWR,
      vmin=-np.max(u_gtr[idx_traj, idx_time, ..., ivar]),
      vmax=np.max(u_gtr[idx_traj, idx_time, ..., ivar]),
    )
    plt.colorbar(h, ax=axs[ivar, :2], fraction=.05)
    h = axs[ivar, 2].imshow(
      np.abs(u_err[idx_traj, idx_time, ..., ivar]),
      cmap=CMAP_WRB,
      vmin=0,
      vmax=np.max(np.abs(u_err[idx_traj, idx_time, ..., ivar])),
    )
    plt.colorbar(h, ax=axs[ivar, 2], fraction=.1)

  axs[0, 0].set(title='Ground-truth');
  axs[0, 1].set(title='Estimate');
  axs[0, 2].set(title='Absolute error');

  for ivar in range(n_vars):
    axs[ivar, 0].set(ylabel=f'Variable {ivar:02d}');

  return fig, axs

def animate_estimations(u_gtr, u_prd, u_err, idx_traj=0):

  n_vars = u_gtr.shape[-1]
  n_time = u_gtr.shape[1]
  fig, axs = plt.subplots(nrows=n_vars, ncols=3, figsize=(5*n_vars, 3*n_vars), sharex=True, sharey=True)

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
    plt.colorbar(h, ax=axs[ivar, :2], fraction=.05)
    h = axs[ivar, 2].imshow(
      np.abs(u_err[idx_traj, 0, ..., ivar]),
      cmap=CMAP_WRB,
      vmin=0,
      vmax=np.max(np.abs(u_err[idx_traj, :, ..., ivar])),
    )
    handlers_err.append(h)
    plt.colorbar(h, ax=axs[ivar, 2], fraction=.1)

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

  # Because the initial state is removed from the errors
  idx_time=[i-1 for i in idx_time]

  n_vars = u_err.shape[-1]
  fig, axs = plt.subplots(nrows=n_vars, ncols=len(idx_time), figsize=(10, 1.5*n_vars+.2), sharex=True, sharey=True)

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
