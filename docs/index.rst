=======================================
A Collection of Fashion Outfit Datasets
=======================================

Introduction
============

Welcome to our guide on utilizing the fashion outfit datasets for recommendations. We differentiate between non-personalized and personalized datasets, with the latter linking each outfit to a specific user for tailored recommendations. Essentially, non-personalized recommendations can be viewed as a special case of personalized recommendations with a single universal user. Imagine an outfit as a collection of :math:`n` items from diverse categories, represented as :math:`\mathcal{O}=\{x_1, \ldots, x_n\}`, where each outfit is associated with a user index :math:`u` from the set :math:`\mathcal{U}=\{1, \ldots, m\}`. This document outlines the dataset's structure and demonstrates how to efficiently load and use it.

Data Encoding
=============

Fashion items
-------------
Fashion items are categorized into different types, such as tops, bottoms, shoes, and accessories. We use ``item_list`` to store a map for items across categories, where ``item_list[c]`` encompasses items within the :math:`c`-th category. Each item's key within ``item_list[c]`` is essential for data loading.

.. code-block:: python

   item_list = [
      [item_key_1_1, item_key_1_2, ...], # category 1
      [item_key_2_1, item_key_2_2, ...], # category 2
      # ...
      [item_key_n_1, item_key_n_2, ...]  # category n
   ]

With ``item_list``, we can encode each item as a tuple: :math:`(c, n)`, where :math:`c` is the category index and :math:`n` is the item index within the category. This abstraction allows to generate new outfits efficiently. ``item_list`` is usually used globally across different splits.

Fashion outfits
---------------

Outfits are encoded as follows:

.. code-block:: python

   data = [uid, size, *items, *types]

- ``uid``: Interger. User index.
- ``size``: Interger. Outfit length.
- ``items``: List of intergers. Indexes of items. 
- ``types``: List of intergers. Categories of items.


``items`` and ``types`` are set to a predefined ``max_size``. If an outfit's size is below ``max_size``, we append ``-1`` to indicate absent items. For example, ``[0, 3, 1, 4, 6, -1, 0, 1, 2, -1]`` represents an outfit with user index ``0``, size ``3``, and items :math:`(i_{01}, i_{14}, i_{26})` from categories 0, 1, 2, respectively. 

To find the corresponding unique key for each item, we can use the following code:

.. code-block:: python
   
   max_size = len(outfit) // 2 - 1
   uid, size, outfit = data[:2], data[2:]
   items, types = outfit[:max_size], outfit[max_size:]
   for i in range(size):
      c = types[i]
      n = items[i]
      print(f"Category: {c}, Item key: {item_list[c][n]}")

With such encoding, each split is simply an array:

.. code-block:: python

   train_data # shape (n_train, max_size*2+2)
   val_data   # shape (n_val,   max_size*2+2)
   test_data  # shape (n_test,  max_size*2+2)

And it is the basic format for outfit generation.

Outfit Generation
=================

.. currentmodule:: outfit_datasets.generator

Given the above tuple format, we can easily generate new tuples with different subclasses of :class:`Generator`:

- ``generator = Generator(init_data=None, **kwargs)``: the interface for all generators.
- ``generator(input_data)``: generate tuples with given input data.

Supported types of generators are:

- :class:`Fix` always returns ``init_data``, ingore the ``input_data`` during each call.
- :class:`Identity` always returns ``input_data``.
- :class:`RandomMix` returns randomly mixed tuples of ``input_data``.
- :class:`RandomReplace` randomly replace :math:`k` items in ``input_data``.
- :class:`FITB` randomly replace one item in ``input_data`` for FITB task.

Outfit Datadaset
----------------
.. currentmodule:: outfit_datasets


Basic Dataset
-------------


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
-----------------


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
