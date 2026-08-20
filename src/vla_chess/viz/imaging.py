"""PNG output and caption/panel composition for the repo's committed reference images."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

# 5x7 bitmap glyphs, drawn at 2x. A caption burned into the image is what keeps a reference
# render self-identifying once it is out of this repo and in a chat window.
_GLYPHS: Final = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "a": ("00000", "00000", "01110", "00001", "01111", "10001", "01111"),
    "b": ("10000", "10000", "10110", "11001", "10001", "10001", "11110"),
    "c": ("00000", "00000", "01110", "10001", "10000", "10001", "01110"),
    "d": ("00001", "00001", "01101", "10011", "10001", "10001", "01111"),
    "e": ("00000", "00000", "01110", "10001", "11111", "10000", "01110"),
    "f": ("00110", "01001", "01000", "11100", "01000", "01000", "01000"),
    "g": ("00000", "00000", "01111", "10001", "01111", "00001", "01110"),
    "h": ("10000", "10000", "10110", "11001", "10001", "10001", "10001"),
    "i": ("00100", "00000", "01100", "00100", "00100", "00100", "01110"),
    "j": ("00010", "00000", "00110", "00010", "00010", "10010", "01100"),
    "k": ("10000", "10000", "10010", "10100", "11000", "10100", "10010"),
    "l": ("01100", "00100", "00100", "00100", "00100", "00100", "01110"),
    "m": ("00000", "00000", "11010", "10101", "10101", "10101", "10101"),
    "n": ("00000", "00000", "10110", "11001", "10001", "10001", "10001"),
    "o": ("00000", "00000", "01110", "10001", "10001", "10001", "01110"),
    "p": ("00000", "00000", "11110", "10001", "11110", "10000", "10000"),
    "q": ("00000", "00000", "01111", "10001", "01111", "00001", "00001"),
    "r": ("00000", "00000", "10110", "11001", "10000", "10000", "10000"),
    "s": ("00000", "00000", "01111", "10000", "01110", "00001", "11110"),
    "t": ("01000", "01000", "11110", "01000", "01000", "01001", "00110"),
    "u": ("00000", "00000", "10001", "10001", "10001", "10011", "01101"),
    "v": ("00000", "00000", "10001", "10001", "10001", "01010", "00100"),
    "w": ("00000", "00000", "10001", "10001", "10101", "10101", "01010"),
    "x": ("00000", "00000", "10001", "01010", "00100", "01010", "10001"),
    "y": ("00000", "00000", "10001", "10001", "01111", "00001", "01110"),
    "z": ("00000", "00000", "11111", "00010", "00100", "01000", "11111"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "=": ("00000", "00000", "11111", "00000", "11111", "00000", "00000"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    " ": ("00000",) * 7,
}


def write_png(path: Path, image: npt.NDArray[np.uint8]) -> None:
    """Write an image as an RGB PNG, using only the standard library.

    Args:
        path: Destination file, overwritten if it exists.
        image: Pixels as (height, width, 3) in RGB order.
    """
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        """Frame a payload as a length-tag-data-CRC PNG chunk."""
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def caption(width: int, text: str, height: int = 22) -> npt.NDArray[np.uint8]:
    """Draw a caption bar to sit above a panel.

    Args:
        width: Bar width in pixels; matches the panel it labels.
        text: Caption text. Characters outside the bitmap font render as spaces.
        height: Bar height in pixels.

    Returns:
        A black strip carrying the text in white, as (height, width, 3) RGB.
    """
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    x = 4
    for character in text:
        glyph = _GLYPHS.get(character, _GLYPHS[" "])
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit == "1":
                    bar[6 + row * 2 : 8 + row * 2, x + column * 2 : x + column * 2 + 2] = 255
        x += 12
    return bar


def downscale(image: npt.NDArray[np.uint8], factor: int) -> npt.NDArray[np.uint8]:
    """Shrink an image by an integer factor, averaging each block.

    Args:
        image: Pixels as (height, width, 3) in RGB order.
        factor: Side length in pixels of the block averaged into one output pixel. Rows and
            columns past the last whole block are dropped.

    Returns:
        The averaged image, with both spatial dimensions divided by the factor. The input
        itself when the factor is 1.

    Raises:
        ValueError: If the factor is below 1.
    """
    if factor < 1:
        raise ValueError(f"downscale factor must be at least 1, got {factor}")
    if factor == 1:
        return image
    height, width, channels = image.shape
    trimmed = image[: height - height % factor, : width - width % factor]
    blocks = trimmed.reshape(
        trimmed.shape[0] // factor, factor, trimmed.shape[1] // factor, factor, channels
    )
    return np.asarray(blocks.mean(axis=(1, 3)).round().astype(np.uint8))


def contact_sheet(
    panels: Sequence[tuple[str, npt.NDArray[np.uint8]]], columns: int
) -> npt.NDArray[np.uint8]:
    """Lay captioned panels out in a grid, padding the last row to full width.

    Args:
        panels: Caption and image pairs, placed left to right then top to bottom. All images
            must share a shape, since the grid is stacked rather than resized.
        columns: Panels per row. The last row is filled out with blank tiles.

    Returns:
        The assembled sheet, each panel carrying its caption bar above it.

    Raises:
        ValueError: If there are no panels, or fewer than one column.
    """
    if not panels:
        raise ValueError("a contact sheet needs at least one panel")
    if columns < 1:
        raise ValueError(f"columns must be at least 1, got {columns}")
    tiles = [np.vstack([caption(image.shape[1], text), image]) for text, image in panels]
    blank = np.zeros_like(tiles[0])
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        row += [blank] * (columns - len(row))
        rows.append(np.hstack(row))
    return np.ascontiguousarray(np.vstack(rows))
