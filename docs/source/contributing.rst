Contributing
============

We welcome contributions to lisa-gap! Whether you're reporting bugs, suggesting features, improving documentation, or contributing code, your help is appreciated.

Getting Started
---------------

If you're interested in contributing, here are several ways you can help:

Reporting Issues
~~~~~~~~~~~~~~~~

Found a bug or have a feature request? Please check if it already exists in our `issue tracker <https://github.com/ollieburke/lisa-gap/issues>`_ and create a new issue if needed.

When reporting bugs, please include:

* Your operating system and Python version
* lisa-gap version
* A minimal code example that reproduces the issue
* The full error message or unexpected behavior

Suggesting Features
~~~~~~~~~~~~~~~~~~~

We're always interested in new ideas! Open an issue with the "enhancement" label and describe:

* What you'd like to see
* Why it would be useful
* How you envision it working

Contributing Code
-----------------

If you'd like to contribute code, here's how to get started:

Setting Up Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/yourusername/lisa-gap.git
      cd lisa-gap

3. Install in development mode:

   .. code-block:: bash

      pip install -e ".[dev]"

4. Create a new branch for your work:

   .. code-block:: bash

      git checkout -b feature-name

Making Changes
~~~~~~~~~~~~~~

* Write clear, documented code
* Add tests for new functionality
* Ensure existing tests still pass
* Follow the existing code style
* Update documentation as needed

Testing
~~~~~~~

Run the test suite to ensure your changes don't break existing functionality:

.. code-block:: bash

   pytest

Submitting Changes
~~~~~~~~~~~~~~~~~~

1. Commit your changes with a clear commit message
2. Push to your fork on GitHub
3. Open a pull request against the main repository

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

* Provide a clear description of what your changes do
* Reference any related issues
* Include tests for new functionality
* Ensure all tests pass
* Update documentation if needed

Documentation
-------------

Documentation improvements are always welcome! You can:

* Fix typos or unclear explanations
* Add examples or tutorials
* Improve API documentation
* Translate documentation (future feature)

To build documentation locally:

.. code-block:: bash

   cd docs
   make html

Questions?
----------

If you have questions about contributing, feel free to:

* Open an issue for discussion
* Reach out to the maintainers
* Start a discussion on GitHub
* email Ollie Burke directly -- ollie.burke@glasgow.ac.uk

Happy coding! 
