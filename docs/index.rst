A Collection of Fashion Outfit Datasets
=======================================

Introduction
------------
In this tutorial, we will show you how to use current released fashion outfits datasets.
The different between non-personalized and personalized fashion outfits datasets is that each outfit is associated with a user in personalized recommendation.
Technically, the non-personalized outfit recommendation can be seen as personalized outfit recommendation where there is only one user.
Suppose that an outfit consists of a set of :math:`n` items from different categories, e.g. :math:`\mathcal{O}=\{x_1, \ldots, x_n\}`.
Each outfit is associated with a user index :math:`u`, where :math:`u\in\mathcal{U}=\{1, \ldots, m\}`.
In the following sections, we define the format of the dataset and show how to load the dataset.

Outfit tuple format
~~~~~~~~~~~~~~~~~~~
We first define a list of items by ``item_list``, where ``item_list[i]`` is the list of items in the :math:`i`-th category.
``item_list[i]`` saves the key of each item for loading the features, and the each outfit is conveterd using the item index in it.
Overall, we represent each item as the index in the corresponding category and outfit with following format:

.. code-block::

   [uid, size, [items], [types]]

where:

- ``uid``: the user id :math:`u`.
- ``size``: length of outfit.
- ``[items]``: item indexes in the corresponding categories.
- ``[types]``: categories for each item in ``[items]``.


   The size of ``items`` and ``types`` equals to a pre-defined value ``max_size``. If the size of an outfit is less than ``max_size``, ``-1`` is appended to represent the non-existence.


Examples

.. code-block:: python

   print(item_list)
   # key for items in different categories
   # [
   #    [item_key1, item_key2, ...],
   #    [item_key1, item_key2, ...],
   #    ...
   # ]
   for i, (n, c) in enumerate(zip(items, types)):
       if n == -1:
           break
      print("n-th item: ", item_list[c][n])


Negative tuple generation
~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: outfit_datasets.generator

Given the outfit tuple format, we can easily generate negative tuples with different subclasses of :class:`Generator`:

- ``generator=Generator(init_data, **kwargs)``: the interface for defining a generator with given data
- ``generator(input_data)``: generate tuples with given input data

Supported types of generators are:

- :class:`FixGenerator` always returns ``init_data``.
- :class:`IdentityGenerator` always returns ``input_data``.
- :class:`RandomMixGenerator` returns randomly mixed tuples.
- :class:`RandomReplaceGenerator` randomly replace :math:`k` out of :math:`n` items in outfit.
- :class:`FITBGenerator` randomly generates tuples for FITB task.

Outfit Data Class
-----------------
.. currentmodule:: outfit_datasets


Basic Dataset
~~~~~~~~~~~~~

The :class:`BaseOutfitData` defines different outfit data.
For all datasets, we have the positive tuples and the negative tuples if existed.


- Datum: Given a key, return the data of the corresponding item.
- positive tuples: The :math:`n\times m` array for outfits. Required for all dataset.
- negative tuples: The :math:`n\times m` array for outfits. Optional.
- positive data mode: how to generate positive data, usually fixed.
- positive data param: configuration for specific mode.
- negative data mode: how to generate negative data, usually randomly mixed.
- negative data param: configuration for specific mode.


Outfit DataLoader
~~~~~~~~~~~~~~~~~

A high-level configuration for outfit data. We introduce :class:`OutfitLoader` as a high-level implementation for outfit data.


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Introduction

   intro

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Outfit Datasets

   iqon_3000
   maryland_polyvore
   polyvore_outfits
   polyvore_u
   shift15m


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Modules

   modules
