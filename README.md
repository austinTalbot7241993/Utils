# Utils

This is a repository containing python functions that I ended up repeatedly using and reimplmenting. These functions fall into 2 categories, numpy and tensorflow, which are my primary libraries for scientific computing.

Every function is implemented in the Code directory. Associated with every function and file is a unit test in the UnitTests directory. This ensures that every function is correctly implemented with no mathematical bugs. All functions are commented in the style of sklearn, a single sentence description of the function purpose, a list of inputs with descriptions and types, and a list of outputs. The top of every file contains a summary of the theme of the contained functions.

The general focus of these methods is statistical, a particularly useful class computes relevant quantities for multivariate normals, including conditional distributions and log likelihoods. These are used heavily in my implementation of probabilistic PCA and associated latent variable models.
