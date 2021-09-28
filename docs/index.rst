Outfit Datasets
===============

Outfit data tuple 
-----------------

In personalized outfit recommendation, we have a set of outfits with user information. Technically, the non-personalized outfit recommendation can be seen as personalized outfit recommendation where there is only one user. Suppose that an outfit consists of :math:`n` items from different categories, e.g. :math:`\{x_1, \ldots, x_n\}` and the outfit is liked by the :math:`u`-th user.

Tuple format
~~~~~~~~~~~~

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

Tuple Generation
~~~~~~~~~~~~~~~~

:class:`outfit_datasets.generator.Generator`

Types of generators:

- "Fix": return stored tuples.
- "Identity": return input.
- "RandomMix": reutrn randomly mixed tuples.
- "RandomReplace": randomly replace :math:`n` items in outfit.


Data Reader
-----------

Given a key, return the data of the corresponding item.


Outfit Data
-----------

:class:`outfit_datasets.OutfitData`

Output Format
-------------

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
   polyvore_maryland
   polyvore_outfits
   polyvore_u
   shift15m


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Modules

   outfit_data