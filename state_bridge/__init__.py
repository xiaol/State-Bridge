"""State Bridge: latent hand-off between frozen language models.

A small trained *bridge* maps the hidden states a frozen sender model builds
while reading a prompt into the input space of a frozen receiver model, which
then does all of the generation.  Neither model's weights are touched; the
bridge is the only part of the system that learns.
"""

__version__ = "0.1.0"
