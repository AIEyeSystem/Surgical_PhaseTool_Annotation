Installation
====================================

Option 1: Install from a source checkout
-----------------------------------------

1.  Create a new environment (recommended, optional)

    Use conda create a new environment

    .. code-block:: bash

       conda create -n ivas_env python=3.10

    activate new environment

    .. code-block:: bash

        conda activate ivas_env

2.  Change into the IVAS source directory and install the local package

    .. code-block:: bash

        pip install .

    .. attention::

        To use GPU on Windows OS, install torch and torchvision from `Pytorch <https://pytorch.org/>`_

3.  Run

    Start the application via the command line:

    .. code-block:: bash

        ivas

.. hint:: This installs the source tree you received; it does not assume that
          an ``ivas-sam`` package has already been published to PyPI.

Option 2: Run directly from source
------------------------------------

1.  Obtain the source tree

    Use the source archive or repository address supplied with your IVAS
    distribution, then change into its root directory. This documentation does
    not hard-code an upstream repository as the current IVAS support channel.

2.  Create a new environment (recommended, optional)

    Use conda create a new environment

    .. code-block:: bash

        conda create -n ivas_env python=3.10

    activate new environment

    .. code-block:: bash

        conda activate ivas_env

3.  Install dependencies

    .. code-block:: bash

        pip install -r requirements.txt

    .. attention::

        To use GPU on Windows OS, install torch and torchvision from `Pytorch <https://pytorch.org/>`_

4.  Run

    Start the application via the command line:

    .. code-block:: bash

        python main.py

    Or

    .. code-block:: bash

        # install the current source tree as a package
        pip install .
        # run the installed command
        ivas

