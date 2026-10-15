Welcome to IVAS's documentation!
================================

**IVAS** is a visual annotation toolkit powered by `Segment Anything Model <https://github.com/facebookresearch/segment-anything>`_.

IVAS is maintained by zack zhao. Use the support channel supplied with your
distribution for current source, releases, and feedback.

.. image:: ../../logo/logo.png
    :alt: IVAS logo
    :width: 180px
    :align: center

Features
====================================

.. hint:: Focusing on the field of image segmentation, we strive to create the best image segmentation annotation software.

- Better performance and a smoother annotation experience.
- Supports `SAM <https://github.com/facebookresearch/segment-anything>`_, `SAM2 <https://github.com/facebookresearch/segment-anything-2>`_, `SAM3 <https://github.com/facebookresearch/sam3>`_, `Sam-HQ <https://github.com/SysCV/sam-hq>`_, `MobileSAM <https://github.com/ChaoningZhang/MobileSAM>`_, `EdgeSAM <https://github.com/chongzhou96/EdgeSAM>`_ and `MedSAM <https://github.com/bowang-lab/MedSAM>`_.
- Supports `Segment anything model <https://github.com/facebookresearch/segment-anything>`_ with ``point prompt`` and ``box prompt`` to semi-automatic annotation.
- Supports ``text prompt`` and ``visual prompt`` based on sam3.
- Supports using remote server for SAM encoding.
- Supports unified ``video annotation`` with temporal labels, staged tracking,
  review, transactional commit, and rollback.
- Supports plugin extensions.

Contents
====================================

.. toctree::
   install
   usage
   plugin
   contact_us
   api
