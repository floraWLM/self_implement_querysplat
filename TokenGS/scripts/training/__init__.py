"""Training-only adapters around the unchanged QuerySplat inference graph."""

from .vggt_input_pass import VGGTInputPassOutput, forward_vggt_input_once

__all__ = ["VGGTInputPassOutput", "forward_vggt_input_once"]
