# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a non-instanceable copy of the Unitree Go2 USD asset.

The stock instanceable ``go2.usd`` triggers an Isaac Sim 6.0 render-path issue
(IsaacLab issue 2925 / IsaacSim issue 227): in the play/viewer camera path the visual
meshes expand only partially, so recorded videos and the viewport show a fragmented
robot (body and some legs missing). Flattening the composed stage (inlining the
referenced ``Props/instanceable_meshes.usd``) and clearing the ``instanceable`` flag on
every prim makes each link carry its own visual geometry, which renders reliably.

The play script :mod:`scripts/reinforcement_learning/rsl_rl/play.py` swaps in the output
of this tool for the visualization path only; training keeps the instanceable asset,
which is memory-efficient at large environment counts.

The output lands under ``ISAACLAB_ASSETS_DATA_DIR/Robots/Go2_noninstanceable/go2.usd``,
which is where the play script looks for it. This asset is not tracked by git, so re-run
this tool once per checkout/machine.

Run with::

    ./isaaclab.sh -p scripts/tools/make_go2_noninstanceable.py

The source asset defaults to the stock ``ISAACLAB_NUCLEUS_DIR`` Go2 asset; override with
``--source`` if it lives elsewhere.
"""

import argparse
import os

from pxr import Usd

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR


def main() -> None:
    default_source = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd"
    default_dest = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Robots", "Go2_noninstanceable", "go2.usd")

    parser = argparse.ArgumentParser(description="Create a non-instanceable copy of go2.usd.")
    parser.add_argument("--source", default=default_source, help="Path/URL to the source (instanceable) go2.usd.")
    parser.add_argument("--dest", default=default_dest, help="Output path for the non-instanceable asset.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.dest), exist_ok=True)

    print(f"Opening source: {args.source}")
    stage = Usd.Stage.Open(args.source)
    if stage is None:
        raise RuntimeError(
            f"Could not open source asset: {args.source}. If it is a remote (S3/Nucleus) URL, pass a local copy"
            " via --source."
        )

    # Flatten composes all references/payloads (incl. Props/instanceable_meshes.usd) into a
    # single self-contained root layer.
    print("Flattening composed stage (inlining referenced meshes)...")
    flat_layer = stage.Flatten()
    flat_stage = Usd.Stage.Open(flat_layer)

    # Clear the instanceable flag everywhere so the visual geometry is expanded per-prim.
    count = 0
    for prim in flat_stage.Traverse():
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
            count += 1
    print(f"Cleared instanceable flag on {count} prim(s).")

    flat_stage.GetRootLayer().Export(args.dest)
    print(f"Exported non-instanceable asset to: {args.dest}")
    print(f"Output size: {os.path.getsize(args.dest) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
