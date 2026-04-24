#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:39:08 2020.

@author: spiros
"""
import numpy as np


def nb_vals(matrix, indices, size=1, perimeter=False):
    """
    Extract the indices of neighboring elements around a specific center in a matrix.

    This function utilizes L_inf (Chebyshev) distance bounding boxes for highly
    optimized, O(1) local coordinate extraction, replacing the need for O(M*N)
    global distance transforms.

    Ref:
    https://stackoverflow.com/questions/49210506/how-to-get-all-the-values-of-neighbours-around-an-element-in-matrix

    Parameters
    ----------
    matrix : numpy.ndarray
        The input matrix. Used strictly to determine spatial boundaries (M, N).
    indices : list or tuple
        The (row, col) coordinates representing the center of the neighborhood.
    size : int, optional
        The radius (Chebyshev distance) of the neighborhood. The default is 1.
    perimeter : boolean, optional
        If set true, the function returns only the exact perimeter of
        the neighborhood (distance == size). Otherwise, it returns the entire
        filled neighborhood (distance <= size). The default is False.

    Returns
    -------
    nb_indices : numpy.ndarray
        A 2D array of shape (num_points, 2) containing the spatial coordinates
        of the requested neighborhood, safely clamped to the image boundaries.
    """
    M, N = matrix.shape
    # Handle the case where indices might be passed as a nested list or tuple
    r = int(np.atleast_1d(indices)[0])
    c = int(np.atleast_1d(indices)[1])

    # Define the bounding box, clamped strictly to image boundaries
    r_min = max(0, r - size)
    r_max = min(M - 1, r + size)
    c_min = max(0, c - size)
    c_max = min(N - 1, c + size)

    # Generate a local 2D grid for just this small patch
    rr, cc = np.meshgrid(
        np.arange(r_min, r_max + 1),
        np.arange(c_min, c_max + 1),
        indexing='ij'
    )

    if perimeter:
        # Chebyshev distance formula: max(|x1 - x2|, |y1 - y2|)
        # We only keep coordinates where the distance is exactly `size` (the perimeter)
        dist = np.maximum(np.abs(rr - r), np.abs(cc - c))
        mask = (dist == size)

        # Extract only the perimeter coordinates
        return np.column_stack((rr[mask], cc[mask]))

    else:
        # We want the entire neighborhood (distance <= size)
        return np.column_stack((rr.flatten(), cc.flatten()))


def random_connectivity(inputs, outputs, opt='random', conns=None, rng=None):
    """
    Connectivity matrix between two layers.

    Parameters
    ----------
    inputs : int
        Number of input nodes.
    outputs : int
        Number of output nodes.
    opt : str, optional
        Method of randomness. The default is 'random'.
    conns : int, optional
        Explicit set of number of connections. The default is None.
    rng : int, optional
        A seed to initialize the BitGenerator. The default is None.

    Raises
    ------
    ValueError
        conns positive int number < inputs*outputs.

    Returns
    -------
    numpy.ndarray
        The connectivity matrix.

    """
    # set the random Generator
    if rng is None:
        rng = np.random.default_rng()  # Fallback if none provided

    mask = np.zeros(shape=(inputs, outputs))
    if opt == 'one_to_one':
        idxs = rng.integer(
            low=0,
            high=mask.shape[0],
            size=mask.shape[1]
        )
        for i in range(mask.shape[1]):
            mask[idxs[i], i] = 1

    elif opt == 'random':
        if conns is None or conns <= 0 or not isinstance(conns, int):
            raise ValueError('Specify `conns` as positive integer. '
                             '`conns` was `None` or negative or float')
        elif conns > mask.size:
            raise ValueError('Specify `conns` as positive integer lower '
                             'than `inputs*outputs`')
        # nodes receive a random number of connections
        indices = rng.choice(inputs*outputs, conns, replace=False)
        mask.flat[indices] = 1

    elif opt == 'constant':
        if conns is None or conns <= 0 or not isinstance(conns, int):
            raise ValueError('Specify `conns` as positive integer. '
                             '`conns` was `None` or negative or float')
        if conns > mask.shape[0]:
            raise ValueError('`conns` cannot be more than input nodes.')
        # all nodes receive the same number of connections
        for i in range(mask.shape[1]):
            idx = rng.choice(mask.shape[0], conns, replace=False)
            mask[idx, i] = 1
    else:
        raise ValueError('Not a valid option. `opt` should take the values'
                         '`one_to_one`, `random` or `constant`')

    return mask.astype('int')


def allocate_synapses(nb, matrix, num_of_synapses, num_channels=1, rng=None):
    """
    The allocation of synapses on dendrites.

    Parameters
    ----------
    nb : list
        List of tuples with all pixels in neighborhood (w, h).
    matrix : TYPE
        DESCRIPTION.
    num_of_synapses : int
        The number of inputs per dendrite.
    num_channels : int, optional
        The number of channels of input images. The default is 1.
    rng : object, optional
        A seed to initialize the BitGenerator. The default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    numpy.ndarray
        The connectivity of one dendrite (each receptive field).

    """
    # set the random Generator
    if rng is None:
        rng = np.random.default_rng()  # Fallback if none provided

    # Allocate inputs to synapses
    M, N = matrix.shape
    mask = np.zeros((M, N))

    # Base neighborhood (radius 1 or 3x3 pixels)
    syn_indices = nb_vals(matrix, list(nb))

    # Expand neighborhood if we don't have enough pixels (e.g., 16 synapses in 3x3)
    if len(syn_indices) < num_of_synapses:
        current_radius = 2

        while len(syn_indices) < num_of_synapses:
            # Grab the next concentric ring (perimeter only)
            extra_syns = nb_vals(matrix, list(nb), size=current_radius, perimeter=True)

            if len(extra_syns) == 0:
                # Safety break: We expanded beyond the whole image
                break
            diff = num_of_synapses - len(syn_indices)

            if len(extra_syns) > diff:
                # We found more than enough in this ring, sample exactly what we need
                chosen_idx = rng.choice(len(extra_syns), size=diff, replace=False)
                syn_indices = np.concatenate((syn_indices, extra_syns[chosen_idx]))
            else:
                # Take all of them and expand again
                syn_indices = np.concatenate((syn_indices, extra_syns))

            current_radius += 1

    # Subsample if we have too many pixels
    elif len(syn_indices) > num_of_synapses:
        idx = rng.choice(len(syn_indices), size=num_of_synapses, replace=False)
        syn_indices = syn_indices[idx]

    # Final safety check
    if len(syn_indices) != num_of_synapses:
        raise ValueError(f"Could not find {num_of_synapses} pixels. Image might be too small!")

    # Vectorized indexing to apply the mask
    row_indices = syn_indices[:, 0]
    col_indices = syn_indices[:, 1]
    mask[row_indices, col_indices] = 1

    if num_channels > 1:
        mask = np.expand_dims(mask, axis=2)
        mask = np.tile(mask, (1, 1, num_channels))

    return mask.reshape(M * N * num_channels)


def make_mask_matrix(
    centers_ids, matrix, dendrites, somata,
    num_of_synapses, num_channels=1,
    rfs_type='somatic', rng=None
    ):
    """
    Create the maks.

    Parameters
    ----------
    centers_ids : list
        The centroids of the RFs.
    matrix : numpy.ndarray
        Zero helper matrix.
    somata : int
        Number of somata.
    dendrites : int
        Number of dendrites per soma.
    num_of_synapses : int
        The number of inputs per dendrite.
    num_channels : int, optional
        The number of channels of input images. The default is 1.
    rfs_type : str, optional
        Type of receptive fields. local (`dendritic`) or global (`somatic`).
        The default is 'somatic'.
    rng : object, optional
        DESCRIPTION.

    Returns
    -------
    mask_final : numpy.ndarray.
        The connectivity matrix.

    """
    if rng is None:
        rng = np.random.default_rng()  # Fallback if none provided

    M, N = matrix.shape
    mask_final = np.zeros((dendrites * somata, matrix.size * num_channels))
    counter = 0

    if rfs_type == 'somatic':
        # Loop for each soma with center --> center of the receptive field
        for center in centers_ids:

            # Base neighborhood (radius 1)
            nb_indices = nb_vals(matrix, list(center), size=1)

            # Expand if we need more dendrites than available pixels
            if dendrites < len(nb_indices):
                current_radius = 2

                while len(nb_indices) < dendrites:
                    extra_centers = nb_vals(matrix, list(center), size=current_radius, perimeter=True)
                    if len(extra_centers) == 0:
                        break # Safety boundary break
                    diff = dendrites - len(nb_indices)
                    if len(extra_centers) > diff:
                        chosen_idx = rng.choice(len(extra_centers), size=diff, replace=False)
                        nb_indices = np.concatenate((nb_indices, extra_centers[chosen_idx]))
                    else:
                        nb_indices = np.concatenate((nb_indices, extra_centers))

                    current_radius += 1

            # Subsample if we have an abundance of pixels
            if len(nb_indices) > dendrites:
                chosen_idx = rng.choice(len(nb_indices), size=dendrites, replace=False)
                nb_indices = nb_indices[chosen_idx]

            for nb in nb_indices:
                mask_final[counter, :] = allocate_synapses(
                    nb, matrix, num_of_synapses,
                    num_channels=num_channels, rng=rng
                )
                counter += 1


    elif rfs_type == 'dendritic':
        for center in centers_ids:
            mask_final[counter, :] = allocate_synapses(
                center, matrix, num_of_synapses,
                num_channels=num_channels, rng=rng
            )
            counter += 1

    return mask_final


def receptive_fields(
    matrix, somata, dendrites, num_of_synapses,
    opt='random', rfs_type="somatic", step=None, prob=None,
    num_channels=1, num_rfs=None, centers_ids=None, rng=None
    ):
    """
    Construct Receptive Fields like connectivity.

    Parameters
    ----------
    matrix : numpy.ndarray
        Zero helper matrix.
    somata : int
        Number of somata.
    dendrites : int
        Number of dendrites per soma.
    num_of_synapses : int
        The number of inputs per dendrite.
    opt : str, optional
        Random or semirandom allocation of centroids. The default is 'random'.
    rfs_type : str, optional
        Type of receptive fields. local (`dendritic`) or global (`somatic`).
        The default is 'somatic'.
    step : TYPE, optional
        DESCRIPTION. The default is None.
    prob : TYPE, optional
        DESCRIPTION. The default is None.
    num_channels : TYPE, optional
        DESCRIPTION. The default is 1.
    num_rfs : int, optional
        DESCRIPTION. The default is None.
    centers_ids : list, optional
        DESCRIPTION. The default is None.
    rng : numpy.random.Generator, optional
        A seed to initialize the BitGenerator. The default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    numpy.ndarray
        The connectivity matrix.
    centers_ids : list
        The centroids of RFs.

    """
    if rng is None:
        rng = np.random.default_rng()  # Fallback if none provided
    M, N = matrix.shape

    if rfs_type == 'somatic':
        nodes = somata
    elif rfs_type == 'dendritic':
        nodes = dendrites*somata

    if not centers_ids:
        if opt == 'random':
            # Random allocation
            flat_indices = rng.choice(M * N, size=nodes, replace=True)
            centers_w, centers_h = np.unravel_index(flat_indices, (M, N))
            centers_ids = list(zip(centers_w, centers_h))
        elif opt == 'random_limited':
            if num_rfs is None:
                raise ValueError('`num_rfs` should be defined under `random_limited` '
                                 'and should be a positive integer. '
                                 'Found `None`')
            # Random allocation -- limited sampling (repeat the centers)
            limited_w = rng.integers(0, M, size=num_rfs)
            limited_h = rng.integers(0, N, size=num_rfs)

            chosen_indices = rng.choice(num_rfs, size=nodes, replace=True)
            centers_w = limited_w[chosen_indices]
            centers_h = limited_h[chosen_indices]
            centers_ids = list(zip(centers_w, centers_h))

        elif opt == 'semirandom':
            if prob is None:
                raise ValueError('`prob` should be defined under `semirandom` '
                                 'and should be a positive float in [0,1]. '
                                 'Found `None`')

            p = rng.random(nodes)
            somata1 = np.sum(p > prob)  # outside of attention site
            somata2 = np.sum(p < prob)  # inside attention site

            # image center coordinates
            w1, w2 = M // 4, 3 * M // 4
            h1, h2 = N // 4, 3 * N // 4

            # Inside attention site (Center crop)
            if somata2 > 0:
                in_w = rng.integers(w1, w2, size=somata2)
                in_h = rng.integers(h1, h2, size=somata2)
                centers_ids2 = list(zip(in_w, in_h))
            else:
                centers_ids2 = []

            # Outside attention site (Peripheral crop)
            if somata1 > 0:
                # To efficiently sample the perimeter, we can sample the whole grid
                # and reject points in the center, or build a valid coordinate list:
                w_coords, h_coords = np.meshgrid(np.arange(M), np.arange(N), indexing='ij')
                center_mask = (w_coords >= w1) & (w_coords < w2) & (h_coords >= h1) & (h_coords < h2)
                valid_periphery = np.argwhere(~center_mask)

                chosen_periphery_idx = rng.choice(len(valid_periphery), size=somata1, replace=True)
                out_coords = valid_periphery[chosen_periphery_idx]
                centers_ids1 = list(zip(out_coords[:, 0], out_coords[:, 1]))
            else:
                centers_ids1 = []

            # concatenate the centers
            centers_ids = centers_ids1 + centers_ids2

        elif opt == 'serial':
            if step is None:
                raise ValueError('`step` should be defined under `serial` '
                                 'and should be a positive integer.'
                                 'Found `None`')

            xv, yv = np.meshgrid(
                range(M),
                range(N),
                sparse=False,
                indexing='ij'
            )
            centers_ids = list(zip(xv.flatten()[::step], yv.flatten()[::step]))

    mask_final = make_mask_matrix(
        centers_ids,
        matrix,
        dendrites,
        somata,
        num_of_synapses,
        num_channels,
        rfs_type,
        rng
    )

    return (mask_final.T.astype('int'), centers_ids)


def connectivity(inputs, outputs):
    """
    Structured connectivity between layer i and layer i+1 nodes.

    Parameters
    ----------
    inputs : int
        Number of nodes in i-th layer.
    somata : int
        Number of nodes in (i+1)-th layer.

    Returns
    -------
    connectivity_matrix : numpy.ndarray [int]
        The connectivity matrix between inputs and outputs.

    Raises
    ------
    ValueError
        If inputs or outputs are non-positive, or if inputs are not divisible by outputs.
    """
    if outputs <= 0:
        raise ValueError("Number of outputs must be greater than zero.")
    if inputs <= 0:
        raise ValueError("Number of inputs must be greater than zero.")
    if inputs % outputs != 0:
        raise ValueError("Inputs must be divisible by outputs without a remainder.")

    connectivity_matrix = np.zeros((inputs, outputs), dtype=int)
    in_per_out = inputs // outputs  # nodes per node
    # Fill the connectivity matrix
    for j in range(outputs):
        start_index = in_per_out * j
        end_index = start_index + in_per_out
        connectivity_matrix[start_index:end_index, j] = 1
    return connectivity_matrix
