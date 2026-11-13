"""Vendored inference code for CSD-MT (CVPR 2024) and its BiSeNet face parser.

CSD-MT: https://github.com/Snowfallingplum/CSD-MT — CC BY-NC-SA 4.0 (LICENSE.txt here).
BiSeNet face parsing: https://github.com/zllrunning/face-parsing.PyTorch — MIT.
Only the pieces needed for inference are kept; the dlib crop is replaced by MediaPipe landmarks
in ``makeup_transfer.engine``.
"""
from .modules import Generator  # noqa: F401
from .bisenet import BiSeNet  # noqa: F401
