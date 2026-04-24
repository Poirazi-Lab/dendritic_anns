#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 09:37:20 2021.

@author: spiros
"""

import os
import time
import keras
import struct
import numpy as np

from keras.utils import Progbar

from receptive_fields import connectivity
from receptive_fields import receptive_fields
from receptive_fields import random_connectivity


def get_data(
    validation_split, dtype='mnist', normalize=True,
    add_noise=False, sigma=None, sequential=False,
    batch_size=None, seed=None
    ):
    """
    Get the dataset.

    Parameters
    ----------
    validation_split : float
        Percent of train data to be held for validation.
    dtype : str, optional
        The name of the dataset.
        Valid names: 'mnist', 'fmnist', 'kmnist', 'emnist, 'cifar10'.
        The default is 'mnist'.
    normalize : boolean, optional
        Normalization of data to [0, 1]. Default is True.
    add_noise : boolean, optional
        Add noise to the data. Default is False.
    sigma : float, optional
        The standard deviation of the Gaussian noise.
        It is ignored if `add_noise=False`.
        The default is None.
    sequential : boolean, optional
        To get the data in a sequential manner. Default is False.
    batch_size : int, optional
        The size of the batch for mini-batch gradient descent. Default is None.
    seed : int, optional
        The integer to initialize the RandomGenerator with the default BitGenerator (PCG64).
        The default is None.

    Returns
    -------
    data : dict
        Dictionary with train, validation and test data.
    labels : dict
        Dictionary with train, validation and test labels.
    img_height : int
        The height of the input images.
    img_width : int
        The width of the input images..
    channels : int
        The number of channels of input images. RGB images contain 3 channels.

    """
    rng = np.random.default_rng(seed)

    # Prepare the training dataset.
    if dtype == 'mnist':
        (x_train, y_train),\
            (x_test, y_test) = keras.datasets.mnist.load_data()
    elif dtype == 'fmnist':
        (x_train, y_train),\
            (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    elif dtype == 'cifar10':
        (x_train, y_train),\
            (x_test, y_test) = keras.datasets.cifar10.load_data()
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
    elif dtype == 'cifar100':
        (x_train, y_train),\
            (x_test, y_test) = keras.datasets.cifar100.load_data()
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
    elif dtype == 'emnist':
        dirname = '../DATASETS/EMNIST'
        x_train = load_dataset(f'{dirname}/emnist-balanced-train-images-idx3-ubyte')
        x_test = load_dataset(f'{dirname}/emnist-balanced-test-images-idx3-ubyte')
        y_train = load_label(f'{dirname}/emnist-balanced-train-labels-idx1-ubyte')
        y_test = load_label(f'{dirname}/emnist-balanced-test-labels-idx1-ubyte')
    elif dtype == 'kmnist':
        dirname = '../DATASETS/KMNIST'
        x_train = load_dataset(f'{dirname}/train-images-idx3-ubyte')
        x_test = load_dataset(f'{dirname}/t10k-images-idx3-ubyte')
        y_train = load_label(f'{dirname}/train-labels-idx1-ubyte')
        y_test = load_label(f'{dirname}/t10k-labels-idx1-ubyte')

    if len(x_train.shape) == 3:
        img_height, img_width = x_train.shape[1:]
        channels = 1
    elif len(x_train.shape) > 3:
        img_height, img_width, channels = x_train.shape[1:]

    # Parse numbers as floats
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalization
    x_train = x_train / 255.
    x_test = x_test / 255.

    x_train = np.reshape(x_train, (-1, channels*img_width*img_height))
    x_test = np.reshape(x_test, (-1, channels*img_width*img_height))

    if sequential:
        dataset = sequential_preprocess(
            x_train, y_train, batch_size=batch_size,
            validation_split=validation_split,
            rng=rng,
        )
        x_train = dataset['xtrain']
        y_train = dataset['ytrain']
        x_val = dataset['xval']
        y_val = dataset['yval']

    else:
        # Standard validation split
        # BEST PRACTICE: Shuffle the indices deterministically before splitting!
        # If the dataset is naturally sorted, slicing [-valsize:] biases validation.
        indices = np.arange(x_train.shape[0])
        rng.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]

        valsize = int(validation_split * x_train.shape[0])
        x_val = x_train[-valsize:]
        y_val = y_train[-valsize:]
        x_train = x_train[:-valsize]
        y_train = y_train[:-valsize]

    # Store the data/labels in a dictionary
    data, labels = {}, {}
    data['train'] = x_train
    data['val'] = x_val
    data['test'] = x_test
    labels['train'] = y_train
    labels['val'] = y_val
    labels['test'] = y_test

    if add_noise:
        for key in data.keys():
            pertrubation = rng.normal(
                loc=0.0,
                scale=sigma,
                size=data[key].shape
            )
            data[key] = perturb_array(
                data[key],
                pertrubation
            )

    return data, labels, img_height, img_width, channels

def check_common_member(a, b):
    """
    Find common members of two arrays.

    Parameters
    ----------
    a : np.ndarray or list
        First array or list.
    b : np.ndarray or list
        Second array or list.

    Returns
    -------
    boolean, `True` if there are common members, `False` otherwise.

    """
    a_set = set(a)
    b_set = set(b)
    if len(a_set.intersection(b_set)) > 0:
        return(True)
    return(False)


def sequential_preprocess(
    input_train, target_train, batch_size,
    validation_split, rng=None):
    """
    Appearance of the data in a sequential manner.

    e.g., class 1, ..., class 1, class 2, ...

    Parameters
    ----------
    input_train : np.ndarray
        The train data.
    target_train : np.ndarray
        The train data labels.
    batch_size : int
        The batch size.
    validation_split : float
        Percent of data to be kept for validation.
    seed : int, optional
        A seed to initialize the BitGenerator. The default is None.

    Raises
    ------
    ValueError
        Raise an error if the same datapoint is in validation and train sets.

    Returns
    -------
    dataset : dict
        The sequential dataset (train and validation, data and targets).

    """
    # set the random Generator
    if rng is None:
        rng = np.random.default_rng()

    target_train = target_train.squeeze()
    a, b = np.unique(
        target_train,
        return_counts=True
    )

    val_size = int(
        validation_split*input_train.shape[0]/(batch_size*len(a))
    )

    k1 = b // batch_size
    ktrain = (k1 - val_size)*batch_size
    kval = b - ktrain

    val_set = []
    for i in range(len(a)):
        idx = np.argwhere(target_train == i).squeeze()
        val_set += list(
            rng.choice(
                idx,
                size=kval[i],
                replace=False
            )
        )

    train_set = list(set(list(range(target_train.shape[0]))) - set(val_set))

    if check_common_member(train_set, val_set):
        raise ValueError('Error in indices.')

    x_val = input_train[val_set]
    y_val = target_train[val_set]

    x_train = input_train[train_set]
    y_train = target_train[train_set]
    # sort the training set
    idx = np.argsort(y_train)
    x_train = x_train[idx]
    y_train = y_train[idx]

    dataset = {}
    dataset['xtrain'] = x_train
    dataset['ytrain'] = y_train
    dataset['xval'] = x_val
    dataset['yval'] = y_val

    return dataset


def load_dataset(path_dataset):
    """
    Load a dataset from a binary file.

    Parameters
    ----------
    path_dataset : str
        The file path to the dataset.

    Returns
    -------
    data : numpy.ndarray
        A 3D array containing the dataset, reshaped to (size, nrows, ncols).

    """
    with open(path_dataset,'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.frombuffer(
            f.read(),
            dtype=np.dtype(np.uint8).newbyteorder('>')
        )
        data = data.reshape((size, nrows, ncols))
        return data


def load_label(path_label):
    """
    Load labels from a binary file.

    Parameters
    ----------
    path_label : str
        The file path to the labels.

    Returns
    -------
    label : numpy.ndarray
       A 3D array containing the dataset, reshaped to (size, nrows, ncols).

    """
    with open(path_label,'rb') as f:
        magic, size = struct.unpack('>II', f.read(8))
        label = np.frombuffer(
            f.read(),
            dtype=np.dtype(np.uint8).newbyteorder('>')
        )
        return label



def perturb_array(arr, perturbation, amin=0, amax=1):
    """
    Perturbation function that adds a given pertubation(s) to a given image(s).

    Parameters
    ----------
    arr : numpy.ndarray
        The input array.
    perturbation : numpy.ndarray
        The perturbed array with same dims as `arr`.
    amin : float, optional
        The minimum value. The default is 0.
    amax : float, optional
        The maximum value. The default is 1.

    Returns
    -------
    numpy.ndarray
        The updated, noisy array with same dims as `arr`.

    """
    return np.clip(arr + perturbation, amin, amax)


def make_masks(
    dends, soma, synapses, num_layers, img_width, img_height,
    num_classes=10, channels=1, conventional=False, sparse=False,
    rfs=True, rfs_type='somatic', rfs_mode='random',
    input_sample=None, seed=None
    ):
    """
    Make masks to transform a traditional ANN in a dendritic ANN.

    Parameters
    ----------
    dends : list
        Number of dendrites/soma per layer.
    soma : list
        Number of somata per layer.
    num_layers : int
        Number of dendrosomatic layers.
    img_width : int
        The width of the input images.
    img_height : int
        The height of the input images.
    channels : int
        The number of channels of the input images.
    conventional : boolean
        If the model is of all-to-all type (True) or not (False).
        Default is False.
    sparse : boolean
        If the model is of random (True) or structured (False) sparse
        connections. Default is False.
    rfs : boolean
        If the model is of RFs (True) or random (False) structured connections.
        Default is True.
    rfs_type : str
        Type of RFs; local (`dendritic`) or global (`somatic`).
        Default is `somatic`.
    rfs_mode : str
        Mode of rfs construction. Default is `random`. Other valid options
        are `one_to_one` and `constant`. Refer to receptive_fields.py in
        random_connectivity function for more information.
    input_sample : str, optional
        DESCRIPTION.
    seed : int, optional
        DESCRIPTION.


    Returns
    -------
    Masks : list
        A list with np.ndarrays containing the boolean masks for all layer
        weights and biases.

    """
    rng = np.random.default_rng(seed)
    Masks = []
    for i in range(num_layers):
        if i == 0:
            # first layer --> create a matrix with input dimensions.
            matrix = np.zeros((img_width, img_height))
        else:
            # for the rest dendrosomatic layers the input is a `square` form of
            # the previous layer's somata.
            divisors = [j for j in range(1, soma[i-1] + 1) if soma[i-1] % j == 0]
            ix = len(divisors) // 2
            if len(divisors) % 2 == 0:
                matrix = np.zeros((divisors[ix], divisors[ix - 1]))
            else:
                matrix = np.zeros((divisors[ix], divisors[ix]))

        # when RFs are enabled!
        if rfs:
            Mask_s_d, centers = receptive_fields(
                matrix, somata=soma[i],
                dendrites=dends[i],
                num_of_synapses=synapses,
                opt=rfs_mode,
                rfs_type=rfs_type,
                prob=0.7,
                num_channels=channels if i == 0 else 1,
                rng=rng
            )
        else:
            # if no RFs are enabled use random connectivity (like `sparse`)
            inputs_size = matrix.size
            factor = channels if i == 0 else 1

            # for soma to the next dendrites (if more than two layers)
            Mask_s_d = random_connectivity(
                inputs=inputs_size*factor,
                outputs=soma[i]*dends[i],
                conns=synapses*soma[i]*dends[i],
                rng=rng
            )
        Masks.append(Mask_s_d)
        # create a mask with `ones` for biases
        Masks.append(np.ones((Mask_s_d.shape[1], )).astype('int'))
        # Create structured connectivity if not `sparse`,
        # else random (i.e., sparse).
        if not sparse:
            Mask_d_s = connectivity(
                inputs=dends[i]*soma[i],
                outputs=soma[i]
            )
        else:
            Mask_d_s = random_connectivity(
                inputs=dends[i]*soma[i],
                outputs=soma[i],
                conns=dends[i]*soma[i],
                rng=rng
            )
        # Append the masks
        Masks.append(Mask_d_s)
        # create a mask with `ones` for biases
        Masks.append(np.ones((Mask_d_s.shape[1], )).astype('int'))

    # If vanilla ANN --> re-write the masks with ones
    # for vanilla ANN all-to-all connectivity and RFs
    if conventional:
        if rfs or sparse:
            # vanilla ANN with random, sparse inputs, or RFs
            for i, m in enumerate(Masks):
                # `4` denotes the number of masks per dendrosomatic layer
                # So, elements 0, 4, 8, 12, etc will have the masks defined above.
                # All other layers will have masks filled with ones.
                if i % 4 != 0:
                    Masks[i] = np.ones_like(m)
        else:
            # vanilla ANN; create all masks with `ones`
            for i, m in enumerate(Masks):
                Masks[i] = np.ones_like(m)

    # dendritic or sparse all-to-all
    if input_sample == 'all_to_all':
        for i, m in enumerate(Masks):
            # `4` denotes the number of masks per dendrosomatic layer
            # So, elements 0, 4, 8, 12, etc will take masks filled with ones.
            if i % 4 == 0:
                Masks[i] = np.ones_like(m)

    # Add two masks for the output layer (weights and biases) set to 1.
    Masks.append(np.ones((Masks[-2].shape[1], num_classes)).astype('int'))
    Masks.append(np.ones((num_classes, )).astype('int'))

    return Masks


def get_model_name(
        conventional=False, rfs=False, sparse=False,
        rfs_type='somatic', input_sample=None):
    """
    Get the model's name in str format.

    Parameters
    ----------
    conventional : boolean, optional
        If the model is fully connected or not. The default is False.
    rfs : boolean, optional
        If the model used receptive field-like sampling. The default is False.
    sparse : boolean, optional
        If the model is randomly-sparsed. The default is False.
    rfs_type : str, optional
        The type of receptive fields, either 'dendritic' or 'somatic'.
        The default is 'somatic'.
    input_sample : str, optional
        If the input is 'all-to-all'. The default is None.

    Returns
    -------
    fname_model : str
        The full name of the model.

    """
    if not conventional:
        if not sparse:
            if rfs:
                fname_model = 'dend_ann_local_rfs' if rfs_type == 'dendritic' else 'dend_ann_global_rfs'
            else:
                fname_model = 'dend_ann_random'
            # to be removed or changed
            if input_sample == 'all_to_all':
                fname_model = 'dend_ann_all_to_all'
        else:
            if rfs:
                fname_model = 'sparse_ann_local_rfs' if rfs_type == 'dendritic' else 'sparse_ann_global_rfs'
            else:
                fname_model = 'sparse_ann'
            # to be removed or changed
            if input_sample == 'all_to_all':
                fname_model = 'sparse_ann_all_to_all'
    else:
        if rfs:
            fname_model = 'vanilla_ann_local_rfs' if rfs_type == 'dendritic' else 'vanilla_ann_global_rfs'
        else:
            fname_model = 'vanilla_ann' if not sparse else "vanilla_ann_random"

    return fname_model


def get_model(
        input_shape, num_layers, dends, soma,
        num_classes, fname_model, relu_slope=0.1,
        dropout=False, rate=0.0):
    """
    Buld the model.

    Parameters
    ----------
    input_shape : tuple
        Shape of inputs.
    num_layers : int
        The number of hidden layers.
    dends : list
        Number of dendrites per node for each layer. `len(dends)` MUST be
        equal to `layers`.
    soma : list
        Number of somata for each layer. `len(soma)` MUST be equal to `layers`.
    num_classes : int
        The number of classes.
    fname_model : str
        The model name.
    relu_slope : float
        The negative slope of leaky relu. Default is 0.1
    dropout : boolean
        If a dropout layer will be added after each hidden layer.
    rate : float
        The rate of the dropout in (0,1). Default is 0.2.

    Returns
    -------
    model : keras.src.models.functional.Functional
        The compiled model. Run `model.summary()` to see its properties.
    """
    # Get model
    # Create the input layer
    input_l = keras.Input(
        shape=input_shape,
        name="input"
    )
    # First hidden dendritic and somatic layer
    # Dendritic layer
    dend_l = keras.layers.Dense(
        dends[0]*soma[0],
        name="dend_1"
    )(input_l)
    # Dendritic activation function
    dend_l = keras.layers.ReLU(
        negative_slope=relu_slope,
        name="dend_1_relu"
    )(dend_l)
    if dropout:
        dend_l = keras.layers.Dropout(
            rate=rate,
            name="dend_1_dropout"
        )(dend_l)
    # Somatic layer
    soma_l = keras.layers.Dense(
        soma[0],
        name="soma_1"
    )(dend_l)
    # Somatic activation function
    soma_l = keras.layers.ReLU(
        negative_slope=relu_slope,
        name="soma_1_relu"
    )(soma_l)
    if dropout:
        soma_l = keras.layers.Dropout(
            rate=rate,
            name="soma_1_dropout"
        )(soma_l)

    # For loop for more layers
    for j in range(1, num_layers):
        # Dendritic layer
        dend_l = keras.layers.Dense(
            dends[j]*soma[j],
            name=f"dend_{j+1}"
        )(soma_l)
        # Activation function
        dend_l = keras.layers.ReLU(
            negative_slope=relu_slope,
            name=f"dend_{j+1}_relu"
        )(dend_l)
        if dropout:
            dend_l = keras.layers.Dropout(
                rate=rate,
                name=f"dend_{j+1}_dropout"
            )(dend_l)
        # Somatic layer
        soma_l = keras.layers.Dense(
            soma[j],
            name=f"soma_{j+1}"
        )(dend_l)
        # Activation function
        soma_l = keras.layers.ReLU(
            negative_slope=relu_slope,
            name=f"soma_{j+1}_relu"
        )(soma_l)
        if dropout:
            soma_l = keras.layers.Dropout(
                rate=rate,
                name=f"soma_{j+1}_dropout"
            )(soma_l)

    # Create the output layer
    output_l = keras.layers.Dense(
        num_classes, activation='softmax',
        name="output"
    )(soma_l)

    # Make the model
    model = keras.Model(
        inputs=input_l,
        outputs=output_l,
        name=fname_model
    )

    return model


def custom_train_loop(
        model, loss_fn, optimizer, Masks, batch_size, num_epochs,
        x_train, y_train, x_val, y_val, x_test, y_test,
        shuffle=True, early_stop=False, patience=0
    ):
    """
    Create the custom training loop for better handling and zeroing out gradients based on masks.

    Parameters
    ----------
    model : keras.src.models.functional.Functional
        The untrained model to be trained.
    loss_fn : TYPE
        The loss function.
    optimizer : TYPE
        The optimization algorithm.
    Masks : list
        List with masks for all layers. There are two maks per layer, one for
        weights and one for biases.
    batch_size : int
        The batch size.
    num_epochs : int
        The number of epochs.
    x_train : numpy.ndarray
        Train set data.
    y_train : numpy.ndarray
        Train set labels.
    x_val : numpy.ndarray
        Validation set data.
    y_val : numpy.ndarray
        Validation set labels.
    x_test : numpy.ndarray
        Test set data.
    y_test : numpy.ndarray
        Test set labels.
    shuffle : boolean, optional
        To shuffle the train data before training. The default is True.
    early_stop : boolean, optional
        To add early stopping during training. The default is False.
    patience : int
        Number of epochs with no improvement after which training will be
        stopped. The default is 0.

    Raises
    ------
    ValueError
        Raise an error if the modified gradient list is not the same size as
        the original.

    Returns
    -------
    model : keras.src.models.functional.Functional
        The trained model. Run `model.summary()` to see its properties.
    out : dict
        The output data. Train loss and accuracy, Validation loss and accuracy
        per epoch, and test loss and accuracy.
    """
    import tensorflow as tf
    # Prepare the metrics
    # Accuracy metrics
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    test_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    # Loss metrics
    train_loss_metric = keras.metrics.SparseCategoricalCrossentropy()
    val_loss_metric = keras.metrics.SparseCategoricalCrossentropy()
    test_loss_metric = keras.metrics.SparseCategoricalCrossentropy()

    # List with the losses for progBar
    metrics_names = ['train_loss', 'val_loss']

    # initialize early stop params on train begin
    if early_stop:
        wait = 0
        best_weights = None
        stopped_epoch = 0
        best = float("inf")


    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # Logits for this minibatch
            logits = model(x, training=True)
            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y, logits)
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)
        # Apply the masks to zero out gradients of non existing connections.
        grads_masked = [tf.math.multiply(g, m) for g, m in zip(grads, Masks)]
        # Check that updated gradients shape is the same as gradients.
        if len(grads) != len(grads_masked):
            raise ValueError("Gradients are unequal in size after masking.")
        # Run one step of gradient descent by updating the value of the
        # variables to minimize the loss.
        optimizer.apply_gradients(zip(grads_masked, model.trainable_weights))
        # Update the training metrics.
        train_acc_metric.update_state(y, logits)
        train_loss_metric.update_state(y, logits)
        return loss_value

    @tf.function
    def val_step(x, y):
        val_logits = model(x, training=False)
        loss_value = loss_fn(y, val_logits)
        # Update the val metrics.
        val_acc_metric.update_state(y, val_logits)
        val_loss_metric.update_state(y, val_logits)
        return loss_value

    @tf.function
    def test_step(x, y):
        test_logits = model(x, training=False)
        loss_value = loss_fn(y, test_logits)
        # Update the test metrics.
        test_acc_metric.update_state(y, test_logits)
        test_loss_metric.update_state(y, test_logits)
        return loss_value

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    # Create the datasets for keras loop.
    if shuffle:
        train_dataset = train_dataset.shuffle(
            buffer_size=train_dataset.cardinality(),
            reshuffle_each_iteration=True
        ).batch(batch_size)
    else:
        train_dataset = train_dataset.batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)

    # lists to store the results
    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []
    cnts_ = x_train.shape[0] // batch_size
    progbar_ = cnts_ if x_train.shape[0] % batch_size == 0 else cnts_ + 1
    # Train loop
    for epoch in range(num_epochs):
        print(f"\nepoch {epoch+1}/{num_epochs}")
        progBar = Progbar(
            progbar_,
            stateful_metrics=metrics_names
        )

        start_time = time.time()
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            train_loss_value = train_step(x_batch_train, y_batch_train)
            # update the bar with the train loss
            progBar.update(step, values=[('train_loss', train_loss_value)])

        # store metrics at the end of each train epoch.
        train_acc_list.append(train_acc_metric.result().numpy())
        train_loss_list.append(train_loss_metric.result().numpy())
        # Reset training metrics at the end of each train epoch
        train_acc_metric.reset_state()
        train_loss_metric.reset_state()

        # Run a validation loop at the end of each epoch.
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_step(x_batch_val, y_batch_val)

        # store validation accuracy and loss
        val_acc_list.append(val_acc_metric.result().numpy())
        val_loss_list.append(val_loss_metric.result().numpy())

        # Update progBar with val_loss
        progBar.update(
            progbar_,
            values=[
                ('train_loss', train_loss_list[-1]),
                ('val_loss', val_loss_list[-1])
            ],
            finalize=True
        )

        # Reset val metrics at the end of each val epoch
        val_acc_metric.reset_state()
        val_loss_metric.reset_state()

        print(f"\nTraining acc over epoch: {float(train_acc_list[-1]):.4f}, "
              f"Validation acc over epoch: {float(val_acc_list[-1]):.4f}")
        print(f"\nTime taken for epoch {epoch}: {time.time() - start_time:.2f}s")

        # Eearly stopping: on epoch end
        if early_stop:
            wait += 1
            if np.less(val_loss_list[-1], best):
                best = val_loss_list[-1]
                wait = 0
                # Record the best weights if current results is better (less).
                best_weights = model.get_weights()
            if wait >= patience:
                stopped_epoch = epoch
                print("\nRestoring model weights from the end of the best epoch.")
                model.set_weights(best_weights)
                break

    # Eearly stopping: on train end
    if early_stop:
        if stopped_epoch > 0:
            print(f"\nEpoch {stopped_epoch + 1}: early stopping")

    # Test on test set
    for step, (x_batch_test, y_batch_test) in enumerate(test_dataset):
        test_step(x_batch_test, y_batch_test)

    # store the test accuracy and loss
    test_acc = test_acc_metric.result().numpy()
    test_loss = test_loss_metric.result().numpy()

    # Update progBar with test_loss
    progBar.update(
        progbar_,
        values=[('test_loss', test_loss)],
        finalize=True
    )

    # Reset test metrics before the end of training
    test_acc_metric.reset_state()
    test_loss_metric.reset_state()

    print(f"Test acc: {float(test_acc):.4f} | "
          f"Test loss: {float(test_loss):.4f}")
    print(f"\nTrain, eval total time: {time.time() - start_time:.2f}s")

    # Save the outputs in a dictionary
    out = {}
    out['train_loss'] = train_loss_list
    out['train_acc'] = train_acc_list
    out['val_loss'] = val_loss_list
    out['val_acc'] = val_acc_list
    out['test_acc'] = test_acc
    out['test_loss'] = test_loss
    if early_stop:
        out['stopped'] = stopped_epoch

    return model, out


def custom_train_loop_torch(
        model, loss_fn, optimizer, Masks, batch_size, num_epochs,
        x_train, y_train, x_val, y_val, x_test, y_test,
        shuffle=True, early_stop=False, patience=0, device='cpu'
    ):
    """
    Create the custom training loop for better handling and zeroing out gradients based on masks.

    Parameters
    ----------
    model : keras.src.models.functional.Functional
        The untrained model to be trained.
    loss_fn : TYPE
        The loss function.
    optimizer : TYPE
        The optimization algorithm.
    Masks : list
        List with masks for all layers. There are two maks per layer, one for
        weights and one for biases.
    batch_size : int
        The batch size.
    num_epochs : int
        The number of epochs.
    x_train : numpy.ndarray
        Train set data.
    y_train : numpy.ndarray
        Train set labels.
    x_val : numpy.ndarray
        Validation set data.
    y_val : numpy.ndarray
        Validation set labels.
    x_test : numpy.ndarray
        Test set data.
    y_test : numpy.ndarray
        Test set labels.
    shuffle : boolean, optional
        To shuffle the train data before training. The default is True.
    early_stop : boolean, optional
        To add early stopping during training. The default is False.
    patience : int
        Number of epochs with no improvement after which training will be
        stopped. The default is 0.

    Raises
    ------
    ValueError
        Raise an error if the modified gradient list is not the same size as
        the original.

    Returns
    -------
    model : keras.src.models.functional.Functional
        The trained model. Run `model.summary()` to see its properties.
    out : dict
        The output data. Train loss and accuracy, Validation loss and accuracy
        per epoch, and test loss and accuracy.
    """
    import torch
    os.environ["KERAS_BACKEND"] = "torch"

    # Create torch Datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train).to(device), torch.from_numpy(y_train).to(device)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_val).to(device), torch.from_numpy(y_val).to(device)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_test).to(device), torch.from_numpy(y_test).to(device)
    )

    # Create DataLoaders for the Datasets
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Prepare the metrics
    # Accuracy metrics
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    test_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    # List with the losses for progBar
    metrics_names = ['train_loss', 'val_loss']

    # initialize early stop params on train begin
    if early_stop:
        wait = 0
        best_weights = None
        stopped_epoch = 0
        best = float("inf")

    # lists to store the results
    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []
    cnts_ = x_train.shape[0] // batch_size
    progbar_ = cnts_ if x_train.shape[0] % batch_size == 0 else cnts_ + 1
    # Train loop
    for epoch in range(num_epochs):
        print(f"\nepoch {epoch+1}/{num_epochs}")
        progBar = Progbar(
            progbar_,
            stateful_metrics=metrics_names
        )

        start_time = time.time()
        running_train_loss = 0.0
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataloader):
            # Forward pass
            train_logits = model(x_batch_train, training=True)
            train_loss = loss_fn(y_batch_train, train_logits)
            running_train_loss += train_loss.cpu().detach().numpy()
            # Update the progbar
            progBar.update(step, values=[('train_loss', train_loss)])

            # Backward pass
            model.zero_grad()
            trainable_weights = [v for v in model.trainable_weights]

            # Call torch.Tensor.backward() on the loss to compute gradients
            # for the weights.
            train_loss.backward()
            gradients = [v.value.grad for v in trainable_weights]

            # Modify the gradients with the masks
            gradients_ = [torch.mul(gradients[i], Masks[i]) for i in range(len(gradients))]
            # Check that updated gradients shape is the same as gradients.
            if len(gradients) != len(gradients_):
                raise ValueError("Gradients are unequal in size after masking.")
            # Update weights
            with torch.no_grad():
                optimizer.apply(gradients_, trainable_weights)

            # Update training metric.
            train_acc_metric.update_state(y_batch_train, train_logits)

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print(f"\nTraining acc over epoch: {float(train_acc):.4f}")

        train_acc_list.append(train_acc.cpu().detach().numpy())
        train_loss_list.append(running_train_loss / (step + 1))
        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_state()

        # Run a validation loop at the end of each epoch.
        running_val_loss = 0.0
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataloader):
            val_logits = model(x_batch_val, training=False)
            val_loss = loss_fn(y_batch_val, val_logits)
            running_val_loss += val_loss.cpu().detach().numpy()
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)

        # Display metrics at the end of each epoch.
        val_acc = val_acc_metric.result()
        val_acc_list.append(val_acc.cpu().detach().numpy())
        val_loss_list.append(running_val_loss / (step + 1))
        # Reset training metrics at the end of each epoch
        val_acc_metric.reset_state()

        # Update progBar with val_loss
        progBar.update(
            progbar_,
            values=[
                ('train_loss', train_loss_list[-1]),
                ('val_loss', val_loss_list[-1])
            ],
            finalize=True
        )

        print(f"\nTraining acc over epoch: {float(train_acc_list[-1]):.4f}, "
              f"Validation acc over epoch: {float(val_acc_list[-1]):.4f}")
        print(f"\nTime taken for epoch {epoch}: {time.time() - start_time:.2f}s")

        # Eearly stopping: on epoch end
        if early_stop:
            wait += 1
            if np.less(val_loss_list[-1], best):
                best = val_loss_list[-1]
                wait = 0
                # Record the best weights if current results is better (less).
                best_weights = model.get_weights()
            if wait >= patience:
                stopped_epoch = epoch
                print("\nRestoring model weights from the end of the best epoch.")
                model.set_weights(best_weights)
                break

    # Eearly stopping: on train end
    if early_stop:
        if stopped_epoch > 0:
            print(f"\nEpoch {stopped_epoch + 1}: early stopping")

    # Test on test set
    running_test_loss = 0.0
    for step, (x_batch_test, y_batch_test) in enumerate(test_dataloader):
        test_logits = model(x_batch_test, training=False)
        test_loss = loss_fn(y_batch_test, test_logits)
        running_test_loss += test_loss.cpu().detach().numpy()
        # Update val metrics
        test_acc_metric.update_state(y_batch_test, test_logits)

    # Update progBar with test_loss
    progBar.update(
        progbar_,
        values=[('test_loss', running_test_loss / (step + 1))],
        finalize=True
    )

    test_acc = test_acc_metric.result().cpu().detach().numpy()
    test_loss = running_test_loss / (step + 1)
    test_acc_metric.reset_state()

    print(f"Test acc: {float(test_acc):.4f} | "
          f"Test loss: {float(test_loss):.4f}")
    print(f"\nTrain, eval total time: {time.time() - start_time:.2f}s")

    # Save the outputs in a dictionary
    out = {}
    out['train_loss'] = train_loss_list
    out['train_acc'] = train_acc_list
    out['val_loss'] = val_loss_list
    out['val_acc'] = val_acc_list
    out['test_acc'] = test_acc
    out['test_loss'] = test_loss
    if early_stop:
        out['stopped'] = stopped_epoch

    return model, out


def custom_train_loop_jax(
        model, loss_fn, optimizer, Masks, batch_size, num_epochs,
        x_train, y_train, x_val, y_val, x_test, y_test,
        shuffle=True, early_stop=False, patience=0
    ):
    """
    Create the custom training loop for better handling and zeroing out gradients based on masks.

    Parameters
    ----------
    model : tensorflow.python.keras.engine.functional.Functional
        The model to be trained.
    loss_fn : TYPE
        The loss function.
    optimizer : TYPE
        The optimization algorithm.
    Masks : list
        List with masks for all layers. There are two maks per layer, one for
        weights and one for biases.
    batch_size : int
        The batch size.
    num_epochs : int
        The number of epochs.
    x_train : numpy.ndarray
        Train set data.
    y_train : numpy.ndarray
        Train set labels.
    x_val : numpy.ndarray
        Validation set data.
    y_val : numpy.ndarray
        Validation set labels.
    x_test : numpy.ndarray
        Test set data.
    y_test : numpy.ndarray
        Test set labels.
    shuffle : boolean, optional
        To shuffle the train data before training. The default is True.

    Raises
    ------
    ValueError
        Raise an error if the modified gradient list is not the same size as
        the original.

    Returns
    -------
    model : tensorflow.python.keras.engine.functional.Functional
        The compiled model. Run `model.summary()` to see its properties.
    out : dict
        The output data. Train loss and accuracy, Validation loss and accuracy
        per epoch, and test loss and accuracy.
    """
    import jax
    import tensorflow as tf

    # Prepare the metrics.
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    test_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    metrics_names = ['train_loss', 'val_loss']

    def compute_loss_and_updates(
        trainable_variables,
        non_trainable_variables,
        metric_variables,
        x, y):

        y_pred, non_trainable_variables = model.stateless_call(
            trainable_variables, non_trainable_variables, x
        )
        loss = loss_fn(y, y_pred)
        metric_variables = train_acc_metric.stateless_update_state(
            metric_variables, y, y_pred
        )
        return loss, (non_trainable_variables, metric_variables)


    grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)


    @jax.jit
    def train_step(state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metric_variables,
        ) = state
        x, y = data
        (loss, (non_trainable_variables, metric_variables)), grads = grad_fn(
            trainable_variables, non_trainable_variables, metric_variables, x, y
        )
        # Modify grads and multiply with masks
        grads_masked = [jax.numpy.multiply(g, m) for g, m in zip(grads, Masks)]
        # Check that updated gradients shape is the same as gradients.
        if len(grads) != len(grads_masked):
            raise ValueError("Gradients are unequal in size after masking.")
        trainable_variables, optimizer_variables = optimizer.stateless_apply(
            optimizer_variables, grads_masked, trainable_variables
        )
        # Return updated state
        state = (trainable_variables,
                 non_trainable_variables,
                 optimizer_variables,
                 metric_variables
                 )
        return loss, state

    @jax.jit
    def eval_step(state, data):
        trainable_variables, non_trainable_variables, metric_variables = state
        x, y = data
        y_pred, non_trainable_variables = model.stateless_call(
            trainable_variables, non_trainable_variables, x
        )
        loss = loss_fn(y, y_pred)
        metric_variables = val_acc_metric.stateless_update_state(
            metric_variables, y, y_pred
        )
        return loss, (
            trainable_variables,
            non_trainable_variables,
            metric_variables,
        )


    @jax.jit
    def test_step(state, data):
        trainable_variables, non_trainable_variables, metric_variables = state
        x, y = data
        y_pred, non_trainable_variables = model.stateless_call(
            trainable_variables, non_trainable_variables, x
        )
        loss = loss_fn(y, y_pred)
        metric_variables = test_acc_metric.stateless_update_state(
            metric_variables, y, y_pred
        )
        return loss, (
            trainable_variables,
            non_trainable_variables,
            metric_variables,
        )

    # Build optimizer variables.
    optimizer.build(model.trainable_variables)
    trainable_variables = model.trainable_variables
    non_trainable_variables = model.non_trainable_variables
    optimizer_variables = optimizer.variables

    # Set-up the train, val and test state tuples
    train_metric_variables = train_acc_metric.variables
    train_state = (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
        train_metric_variables,
    )

    # Set-up the Keras data loaders-datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    # Create the datasets for keras loop.
    if shuffle:
        train_dataset = train_dataset.shuffle(buffer_size=train_dataset.cardinality(),
                                              reshuffle_each_iteration=True).batch(batch_size)
    else:
        train_dataset = train_dataset.batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)

    # initialize early stop params on train begin
    if early_stop:
        wait = 0
        best_weights = None
        stopped_epoch = 0
        best = float("inf")


    # Training loop
    # lists to store the results
    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []
    cnts_ = x_train.shape[0] // batch_size
    progbar_ = cnts_ if x_train.shape[0] % batch_size == 0 else cnts_ + 1
    for epoch in range(num_epochs):
        print(f"\nepoch {epoch+1}/{num_epochs}")
        progBar = Progbar(progbar_, stateful_metrics=metrics_names)

        start_time = time.time()

        running_train_loss = 0.0
        # Iterate over the batches of the dataset.
        for step, train_data in enumerate(train_dataset):
            train_data = (train_data[0].numpy(), train_data[1].numpy())
            train_loss, train_state = train_step(train_state, train_data)
            running_train_loss += train_loss
            # Update the progbar
            progBar.update(step, values=[('train_loss', train_loss)])

        _, _, _, metric_variables = train_state
        for variable, value in zip(train_acc_metric.variables, metric_variables):
            variable.assign(value)
        train_acc_list.append(train_acc_metric.result())
        # Calculate the average loss across batches
        train_loss_list.append(running_train_loss / (step + 1))
        # reset the state
        train_acc_metric.reset_state()

        # prepare for validation
        val_metric_variables = val_acc_metric.variables
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metric_variables,
        ) = train_state
        val_state = (
            trainable_variables,
            non_trainable_variables,
            val_metric_variables,
        )

        # Run a validation loop at the end of each epoch.
        running_val_loss = 0.0
        for step, val_data in enumerate(val_dataset):
            val_data = (val_data[0].numpy(), val_data[1].numpy())
            val_loss, val_state = eval_step(val_state, val_data)
            running_val_loss += val_loss

        # Update progBar with val_loss
        values = [('train_loss', train_loss), ('val_loss', running_val_loss / (step + 1))]
        progBar.update(progbar_, values=values,
                       finalize=True)

        _, _, metric_variables = val_state
        for variable, value in zip(val_acc_metric.variables, metric_variables):
            variable.assign(value)
        val_acc_list.append(val_acc_metric.result())

        # calculate and store validation loss and accuracy
        val_loss_list.append(running_val_loss / (step + 1))
        val_acc_metric.reset_state()

        print(f"\nTraining acc over epoch: {float(train_acc_list[-1]):.4f}, "
              f"Validation acc over epoch: {float(val_acc_list[-1]):.4f}")

        print(f"\nTime taken: {time.time() - start_time:.2f}s")

        # Eearly stopping: on epoch end
        if early_stop:
            wait += 1
            if np.less(val_loss_list[-1], best):
                best = val_loss_list[-1]
                wait = 0
                # Record the best weights if current results is better (less).
                best_weights = model.get_weights()
            if wait >= patience:
                stopped_epoch = epoch
                print("\nRestoring model weights from the end of the best epoch.")
                model.set_weights(best_weights)
                break

    # Eearly stopping: on train end
    if early_stop:
        if stopped_epoch > 0:
            print(f"\nEpoch {stopped_epoch + 1}: early stopping")

    test_metric_variables = test_acc_metric.variables
    (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
        metric_variables,
    ) = train_state
    test_state = (
        trainable_variables,
        non_trainable_variables,
        test_metric_variables,
    )

    # Test on test set
    running_test_loss = 0.0
    for step, test_data in enumerate(test_dataset):
        test_data = (test_data[0].numpy(), test_data[1].numpy())
        test_loss, test_state = test_step(test_state, test_data)
        running_test_loss += test_loss

    # Update progBar with test_loss
    values = [('test_loss', running_test_loss / (step + 1))]
    progBar.update(progbar_, values=values, finalize=True)

    _, _, metric_variables = test_state
    for variable, value in zip(test_acc_metric.variables, metric_variables):
        variable.assign(value)
    test_acc_ = test_acc_metric.result()

    test_acc_metric.reset_state()
    print(f"\nTest acc: {test_acc_:.4f}")
    print(f"\n\nTime taken: {time.time() - start_time:.2f}s")

    # Save the outputs in a dictionary
    out = {}
    out['train_loss'] = train_loss_list
    out['train_acc'] = train_acc_list
    out['val_loss'] = val_loss_list
    out['val_acc'] = val_acc_list
    out['test_acc'] = test_acc_
    out['test_loss'] = running_test_loss / (step + 1)

    # Attach the new variable values back to the model.
    trainable_variables, non_trainable_variables, optimizer_variables, _ = train_state
    for variable, value in zip(model.trainable_variables, trainable_variables):
        variable.assign(value)
    for variable, value in zip(model.non_trainable_variables, non_trainable_variables):
        variable.assign(value)

    return model, out
