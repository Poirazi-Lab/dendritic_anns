#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:39:08 2020.

@author: spiros
"""
import numpy as np
import scipy.ndimage as scn


def nb_vals(matrix, indices, size=1, opt=None):
    """
    https://stackoverflow.com/questions/49210506/how-to-get-all-the-values-of-neighbours-around-an-element-in-matrix.

    Parameters
    ----------
    matrix : the input numpy.array.
    indices : list with the center of the neighborhood.

    Returns
    -------
    nb_indices : indices of the neighborhood.

    """
    indices_ = tuple(np.transpose(np.atleast_2d(indices)))
    arr_shape = matrix.shape
    dist = np.ones(arr_shape)
    dist[indices_] = 0
    dist = scn.distance_transform_cdt(dist, metric='chessboard')

    if opt == 'extended':
        nb_indices = np.transpose(np.nonzero(dist == size))
    else:
        nb_indices = np.transpose(np.nonzero(dist <= size))

    return nb_indices


def random_connectivity(inputs, outputs, opt='random', conns=None, seed=None):
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
    seed : int, optional
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
    rng = np.random.default_rng(seed)

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


def choose_centers(possible_values, nodes, seed):
    """
    Choose coordinates in the image.

    Parameters
    ----------
    possible_values : list
        List of possible pixels to allocate.
    nodes : int
        The number of nodes to allocate centers to.
    seed : int, optional
        A seed to initialize the BitGenerator. The default is None.

    Returns
    -------
    numpy.ndarray
        The center of each node in `nodes`.

    """
    # set the random Generator
    rng = np.random.default_rng(seed)

    return rng.choice(possible_values, nodes)


def allocate_synapses(nb, matrix, num_of_synapses, num_channels=1, seed=None):
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
    seed : int, optional
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
    rng = np.random.default_rng(seed)

    # Allocate inputs to synapses
    M, N = matrix.shape
    mask = np.zeros((M, N))

    syn_indices = nb_vals(matrix, list(nb))

    # Extended neighborhood if syns are missing
    if len(syn_indices) < num_of_synapses:
        diff = num_of_synapses - len(syn_indices)
        extra_syns = nb_vals(matrix, nb, size=2, opt='extended')

        cnt = 3
        while diff > len(extra_syns):
            extra_syns_more = nb_vals(
                matrix,
                nb,
                size=cnt,
                opt='extended'
            )
            extra_syns = np.concatenate(
                (extra_syns, extra_syns_more)
            )
            cnt += 1
            if diff <= len(extra_syns):
                break

        added = extra_syns[
            rng.choice(
                extra_syns.shape[0],
                diff,
                replace=False
            )
        ]
        syn_indices_ = np.concatenate((syn_indices, added))
    elif len(syn_indices) > num_of_synapses:
        # Subsample at random
        idx = rng.choice(
            syn_indices.shape[0],
            num_of_synapses,
            replace=False
        )
        syn_indices_ = syn_indices[idx]
    else:
        syn_indices_ = np.copy(syn_indices)

    if len(syn_indices_) != num_of_synapses:
        raise ValueError('Something is wrong!')

    row_indices = [x[0] for x in syn_indices_]
    col_indices = [x[1] for x in syn_indices_]
    mask[row_indices, col_indices] = 1
    if num_channels > 1:
        mask = np.expand_dims(mask, axis=2)
        mask = np.tile(mask, num_channels)

    return mask.reshape(M*N*num_channels)


def make_mask_matrix(
    centers_ids, matrix, dendrites, somata,
    num_of_synapses, num_channels=1,
    rfs_type='somatic', seed=None
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

    Returns
    -------
    mask_final : numpy.ndarray.
        The connectivity matrix.

    """
    rng = np.random.default_rng(seed)
    
    M, N = matrix.shape
    mask_final = np.zeros((dendrites*somata, matrix.size*num_channels))
    counter = 0

    if rfs_type == 'somatic':
        # Loop for each soma with center--> center of the receptive field
        for center in centers_ids:

            # Find the centers of the neighborhood
            nb_indices = nb_vals(matrix, list(center))

            # if dendrites of one soma are less than the size of the neighborhood
            # pick random centers within the neighborhood
            if dendrites < len(nb_indices):
                nb_indices = nb_indices[
                    rng.choice(
                        range(len(nb_indices)),
                        dendrites,
                        replace=False
                    )
                ]
            # if dendrites of one soma are more than the size of the neighborhood
            # choose random centers of an extended neighborhood
            # (+2 pixel from center)
            elif dendrites > len(nb_indices):
                diff = dendrites - len(nb_indices)
                extra_centers = nb_vals(matrix, center, size=2, opt='extended')

                cnt = 3
                while diff > len(extra_centers):
                    extra_centers_more = nb_vals(
                        matrix, center, size=cnt,
                        opt='extended'
                    )
                    extra_centers = np.concatenate(
                        (extra_centers, extra_centers_more)
                    )
                    cnt += 1
                    if diff <= len(extra_centers):
                        break

                added = extra_centers[
                    rng.choice(
                        extra_centers.shape[0],
                        diff,
                        replace=False
                    )
                ]
                nb_indices = np.concatenate((nb_indices, added))

            # Allocate inputs to synapses
            for nb in nb_indices:
                mask_final[counter, :] = allocate_synapses(
                    nb,
                    matrix,
                    num_of_synapses,
                    num_channels=num_channels,
                    seed=seed
                )
                counter += 1

    elif rfs_type == 'dendritic':
        # Loop for each dendrite with center--> center of the receptive field
        # Allocate inputs to synapses
        for center in centers_ids:
            mask_final[counter, :] = allocate_synapses(
                center,
                matrix,
                num_of_synapses,
                num_channels=num_channels
            )
            counter += 1

    return mask_final


def receptive_fields(
    matrix, somata, dendrites, num_of_synapses,
    opt='random', rfs_type="somatic", step=None, prob=None,
    num_channels=1, size_rfs=None, centers_ids=None, seed=None
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
    size_rfs : int, optional
        DESCRIPTION. The default is None.
    centers_ids : list, optional
        DESCRIPTION. The default is None.
    seed : int, optional
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
    rng = np.random.default_rng(seed)
    M, N = matrix.shape

    if rfs_type == 'somatic':
        nodes = somata
    elif rfs_type == 'dendritic':
        nodes = dendrites*somata

    if not centers_ids:
        if opt == 'random':
            # Random allocation
            centers_w = choose_centers(range(M), nodes, seed)
            centers_h = choose_centers(range(N), nodes, seed + 1)
            centers_ids = [(x, y) for x, y in zip(centers_w, centers_h)]
        elif opt == 'random_limited':
            if size_rfs is None:
                raise ValueError('`size_rfs` should be defined under `random_limited` '
                                 'and should be a positive integer. '
                                 'Found `None`')
            # Random allocation -- limited sampling
            rnd_pixels = rng.integer(
                low=0,
                high=matrix.shape[0],
                size=size_rfs
            )
            centers_w = choose_centers(rnd_pixels, nodes, seed)
            centers_h = choose_centers(rnd_pixels, nodes, seed + 1)
            centers_ids = [(x, y) for x, y in zip(centers_w, centers_h)]

        elif opt == 'semirandom':
            if prob is None:
                raise ValueError('`prob` should be defined under `semirandom` '
                                 'and should be a positive float in [0,1]. '
                                 'Found `None`')
            centers_ids1 = None
            centers_ids2 = None
            p = rng.random(nodes)
            somata1 = sum(p > prob)  # outside of attention site
            somata2 = sum(p < prob)  # inside attention site

            # image center coordinates
            w1, w2 = M//2 - M//4, M//2 + M//4
            h1, h2 = N//2 - N//4, N//2 + N//4
            # Random allocation outside of the middle of the image
            centers_w = choose_centers(
                list(range(w1)) + list(range(w2, M)),
                somata1,
                seed,
            )
            centers_h = choose_centers(
                list(range(h1)) + list(range(h2, N)),
                somata1,
                seed,
            )
            centers_ids1 = [(x, y) for x, y in zip(centers_w, centers_h)]

            # Random allocation in the middle of the image
            centers_w = choose_centers(range(w1, w2), somata2, seed)
            centers_h = choose_centers(range(h1, h2), somata2, seed + 1)
            centers_ids2 = [(x, y) for x, y in zip(centers_w, centers_h)]

            if centers_ids1 is not None and centers_ids2 is not None:
                centers_ids = centers_ids1 + centers_ids2
            elif centers_ids1 is None and centers_ids2 is not None:
                centers_ids = centers_ids2
            elif centers_ids1 is not None and centers_ids2 is None:
                centers_ids = centers_ids1

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
            centers_ids = [[i, j] for i, j in zip(list(xv.flatten()),
                                                  list(yv.flatten()))]
            L = len(centers_ids)

            list_of_indices = list(np.arange(start=0, stop=L, step=step))
            centers_ids = [centers_ids[i] for i in range(len(centers_ids))
                           if i in list_of_indices]

    mask_final = make_mask_matrix(
        centers_ids,
        matrix,
        dendrites,
        somata,
        num_of_synapses,
        num_channels,
        rfs_type,
        seed
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

