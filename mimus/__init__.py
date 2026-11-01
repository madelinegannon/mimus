# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""A curious robot arm: tracker presence -> generative behaviour -> RMPflow -> a gated servo bridge.

The Isaac-free parts live here (behaviour, OSC links, config, UI state); `app.py` at the repo root is
the Isaac Sim script that assembles them."""

__version__ = "0.1.0"
