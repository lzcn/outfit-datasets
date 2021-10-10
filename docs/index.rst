A Collection of Fashion Outfit Datasets
=======================================

Introduction
------------

In personalized outfit recommendation, we have a set of outfits with user information.
Technically, the non-personalized outfit recommendation can be seen as personalized outfit recommendation where there is only one user.
Suppose that an outfit consists of :math:`n` items from different categories, e.g. :math:`\{x_1, \ldots, x_n\}` and the outfit is liked by the :math:`u`-th user.

Outfit tuple format
~~~~~~~~~~~~~~~~~~~

We represent each item with an index in the corresponding cateogry and represent an outfit using following format:

.. code-block::

   [uid, size, [items], [types]]

where:

- ``uid``: user id.
- ``size``: length of outfit.
- ``[items]``: item indexes in the corresponding categories.
- ``[types]``: cateogry index for each item.


   The size of ``items`` and ``types`` equals to a pre-defined value ``max_size``. If the size of an outfit is less than ``max_size``, ``-1`` is appended to represent the non-existence of items.

To get the information of an item, we need to create a list ``x``, where ``x[c][i]`` is the information for :math:`i`-th item in the :math:`c`-th fashion category.

Negative tuple generation
~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`outfit_datasets.generator.Generator`

Types of generators:

- "Fix": return stored tuples.
- "Identity": return input.
- "RandomMix": reutrn randomly mixed tuples.
- "RandomReplace": randomly replace :math:`k` out of :math:`n` items in outfit.

Outfit Data Class
-----------------

Take :ref:`Maryland Polyvore Dataset<maryland_polyvore>` for example, 

Basic Dataset
~~~~~~~~~~~~~

For all datasets, we have positive tuples and negative tuples if exists.
We usually needs different output format. For example


- Datum: Given a key, return the data of the corresponding item. 
- positive tuples: the :math:`n\times m` array for outfits.
- negative tuples: the :math:`n\times m` array for outfits.
- positive data mode: how to generate positive data, usually fixed.
- positive data param: configuration for specific mode, usually not use
- negative data mode: how to generate negative data, usually randomly mixed.
- negative data param: configuration for specific mode, e.g 

We use different subclasses of :class:`outfit_datasets.BaseData` as the implementation.


Outfit DataLoader
~~~~~~~~~~~~~~~~~

A high-level configuration for outfit data. We introduce :class:`outfit_datasets.OutfitData` as a high-level implementation for outfit data.


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
