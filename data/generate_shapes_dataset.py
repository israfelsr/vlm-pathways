"""
Generate a controlled shapes dataset for spatial reasoning experiments.

Creates images with two colored shapes on a white background at known positions.
Each sample has a ground-truth spatial relation (left, right, above, below).

Output: HuggingFace dataset saved to disk, compatible with existing experiment pipeline.

Usage:
    python scripts/generate_shapes_dataset.py
    python scripts/generate_shapes_dataset.py --n-samples 500 --output data/shapes_pairs.hf
    python scripts/generate_shapes_dataset.py --image-size 896 --grid 3x3
"""

import argparse
import math
import random
from itertools import combinations
from pathlib import Path

from datasets import Dataset
from PIL import Image, ImageDraw


# --- Shape definitions ---

SHAPES = ["circle", "square", "triangle", "star", "diamond", "pentagon"]
SHAPES_RECOGNITION = [
    "circle",
    "square",
    "triangle",
    "star",
]  # for forced-choice recognition
ANSWER_COLORS = [
    "red",
    "green",
    "yellow",
    "purple",
]  # for attribute-chain forced-choice
ANSWER_SHAPES = [
    "circle",
    "square",
    "triangle",
    "star",
]  # for attribute-shape forced-choice

COLORS = {
    "red": (220, 50, 50),
    "blue": (50, 80, 220),
    "green": (50, 180, 70),
    "yellow": (230, 200, 40),
    "orange": (240, 140, 30),
    "purple": (150, 50, 200),
}

PREPOSITIONS = {"left", "right", "above", "below"}


def draw_shape(draw, shape, center, size, color):
    """Draw a shape centered at (cx, cy) with given size and color."""
    cx, cy = center
    r = size // 2

    if shape == "circle":
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r], fill=color, outline=(0, 0, 0), width=2
        )

    elif shape == "square":
        draw.rectangle(
            [cx - r, cy - r, cx + r, cy + r], fill=color, outline=(0, 0, 0), width=2
        )

    elif shape == "triangle":
        points = [
            (cx, cy - r),  # top
            (cx - r, cy + r),  # bottom-left
            (cx + r, cy + r),  # bottom-right
        ]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)

    elif shape == "star":
        # 5-pointed star
        points = []
        for i in range(10):
            angle = math.radians(i * 36 - 90)
            radius = r if i % 2 == 0 else r * 0.4
            points.append(
                (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
            )
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)

    elif shape == "diamond":
        points = [
            (cx, cy - r),  # top
            (cx + r, cy),  # right
            (cx, cy + r),  # bottom
            (cx - r, cy),  # left
        ]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)

    elif shape == "pentagon":
        points = []
        for i in range(5):
            angle = math.radians(i * 72 - 90)
            points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)


def compute_relation(pos1, pos2):
    """Compute spatial relation of obj1 relative to obj2.

    Returns the primary spatial relation based on the larger offset axis.
    """
    c1x, c1y = pos1
    c2x, c2y = pos2

    dx = c1x - c2x  # positive = obj1 is to the right of obj2
    dy = c1y - c2y  # positive = obj1 is below obj2

    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "below" if dy > 0 else "above"


def get_grid_positions(image_size, grid_rows, grid_cols, shape_size):
    """Compute cell center positions for a grid layout.

    Returns dict mapping (row, col) -> (cx, cy).
    """
    margin = shape_size
    usable_w = image_size - 2 * margin
    usable_h = image_size - 2 * margin

    positions = {}
    for r in range(grid_rows):
        for c in range(grid_cols):
            cx = margin + int(usable_w * (c + 0.5) / grid_cols)
            cy = margin + int(usable_h * (r + 0.5) / grid_rows)
            positions[(r, c)] = (cx, cy)

    return positions


def generate_sample(image_size, grid_positions, shape_size, rng):
    """Generate one sample: image with two shapes and a spatial relation.

    Returns dict with image, objects, preposition.
    """
    # Pick two distinct grid cells
    cells = list(grid_positions.keys())
    cell1, cell2 = rng.sample(cells, 2)
    pos1 = grid_positions[cell1]
    pos2 = grid_positions[cell2]

    # Pick two distinct shape+color combos
    shape1 = rng.choice(SHAPES)
    shape2 = rng.choice(SHAPES)
    color_names = list(COLORS.keys())
    color1_name, color2_name = rng.sample(color_names, 2)
    color1 = COLORS[color1_name]
    color2 = COLORS[color2_name]

    # Add jitter within the cell (up to 20% of shape_size) to avoid perfectly aligned grids
    jitter = shape_size // 5
    pos1 = (
        pos1[0] + rng.randint(-jitter, jitter),
        pos1[1] + rng.randint(-jitter, jitter),
    )
    pos2 = (
        pos2[0] + rng.randint(-jitter, jitter),
        pos2[1] + rng.randint(-jitter, jitter),
    )

    # Create image
    img = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw_shape(draw, shape1, pos1, shape_size, color1)
    draw_shape(draw, shape2, pos2, shape_size, color2)

    # Compute spatial relation
    relation = compute_relation(pos1, pos2)

    # Object names: "red circle", "blue square"
    obj1_name = f"{color1_name} {shape1}"
    obj2_name = f"{color2_name} {shape2}"

    return {
        "image": img,
        "objects": [obj1_name, obj2_name],
        "preposition": relation,
    }


def generate_balanced_dataset(
    n_samples, image_size, grid_spec, shape_size, seed=42, aligned=False
):
    """Generate a balanced dataset with equal counts per preposition.

    Args:
        n_samples: Total number of samples to generate.
        image_size: Image width/height in pixels.
        grid_spec: Tuple (rows, cols) for position grid.
        shape_size: Shape diameter in pixels.
        seed: Random seed.
        aligned: If True, only use axis-aligned pairs (same row for left/right,
                 same column for above/below). No diagonal placements.

    Returns:
        List of sample dicts.
    """
    rng = random.Random(seed)
    grid_rows, grid_cols = grid_spec
    grid_positions = get_grid_positions(image_size, grid_rows, grid_cols, shape_size)

    # Pre-compute which cell pairs give which relation, for balanced sampling
    relation_to_pairs = {p: [] for p in PREPOSITIONS}
    for c1, c2 in combinations(grid_positions.keys(), 2):
        r1, col1 = c1
        r2, col2 = c2

        if aligned:
            # Same row → left/right only; same column → above/below only
            if r1 != r2 and col1 != col2:
                continue  # skip diagonal pairs

        rel = compute_relation(grid_positions[c1], grid_positions[c2])
        relation_to_pairs[rel].append((c1, c2))
        # Also check reverse
        rev_rel = compute_relation(grid_positions[c2], grid_positions[c1])
        relation_to_pairs[rev_rel].append((c2, c1))

    # Check all relations are reachable
    for p, pairs in relation_to_pairs.items():
        if not pairs:
            print(
                f"Warning: no grid cell pairs produce relation '{p}' with grid {grid_spec}"
            )

    samples_per_relation = n_samples // len(PREPOSITIONS)
    remainder = n_samples % len(PREPOSITIONS)

    samples = []
    for i, prep in enumerate(sorted(PREPOSITIONS)):
        target = samples_per_relation + (1 if i < remainder else 0)
        pairs = relation_to_pairs[prep]
        if not pairs:
            continue

        for _ in range(target):
            cell1, cell2 = rng.choice(pairs)
            pos1 = grid_positions[cell1]
            pos2 = grid_positions[cell2]

            # Pick shapes and colors
            shape1 = rng.choice(SHAPES)
            shape2 = rng.choice(SHAPES)
            color_names = list(COLORS.keys())
            c1_name, c2_name = rng.sample(color_names, 2)

            # Jitter — when aligned, only jitter along the placement axis
            jitter = shape_size // 5
            if aligned and prep in ("left", "right"):
                # Horizontal pair: jitter x freely, keep y fixed
                jpos1 = (pos1[0] + rng.randint(-jitter, jitter), pos1[1])
                jpos2 = (pos2[0] + rng.randint(-jitter, jitter), pos2[1])
            elif aligned and prep in ("above", "below"):
                # Vertical pair: jitter y freely, keep x fixed
                jpos1 = (pos1[0], pos1[1] + rng.randint(-jitter, jitter))
                jpos2 = (pos2[0], pos2[1] + rng.randint(-jitter, jitter))
            else:
                jpos1 = (
                    pos1[0] + rng.randint(-jitter, jitter),
                    pos1[1] + rng.randint(-jitter, jitter),
                )
                jpos2 = (
                    pos2[0] + rng.randint(-jitter, jitter),
                    pos2[1] + rng.randint(-jitter, jitter),
                )

            # Verify relation still holds after jitter
            actual_rel = compute_relation(jpos1, jpos2)
            if actual_rel != prep:
                # Undo jitter if it changed the relation
                jpos1, jpos2 = pos1, pos2

            img = Image.new("RGB", (image_size, image_size), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw_shape(draw, shape1, jpos1, shape_size, COLORS[c1_name])
            draw_shape(draw, shape2, jpos2, shape_size, COLORS[c2_name])

            samples.append(
                {
                    "image": img,
                    "objects": [f"{c1_name} {shape1}", f"{c2_name} {shape2}"],
                    "preposition": prep,
                }
            )

    rng.shuffle(samples)
    return samples


def generate_center_paired_dataset(n_pairs, image_size, grid_spec, shape_size, seed=42):
    """Generate paired samples with center layout for cross-object patching.

    One object (obj2, the reference) is always at the grid center.
    The other object (obj1, asked-about) is at a cardinal position.

    Each pair: same objects, obj1 moves to a different cardinal position → relation changes.
    The center object's image patches are identical between paired images (same visual
    content, same position), enabling clean cross-object patching experiments.

    Args:
        n_pairs: Number of pairs to generate.
        image_size: Image width/height in pixels.
        grid_spec: Tuple (rows, cols) for position grid. Should have odd dimensions for center.
        shape_size: Shape diameter in pixels.
        seed: Random seed.

    Returns:
        List of sample dicts with extra fields: obj1_position, obj2_position,
        shape_size, image_size. Samples at 2*i and 2*i+1 form a pair.
    """
    rng = random.Random(seed)
    grid_rows, grid_cols = grid_spec
    grid_positions = get_grid_positions(image_size, grid_rows, grid_cols, shape_size)

    # Center cell
    center_row = grid_rows // 2
    center_col = grid_cols // 2
    center_cell = (center_row, center_col)
    center_pos = grid_positions[center_cell]

    # Cardinal neighbors only (no diagonals)
    cardinal_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    cardinal_cells = []
    for dr, dc in cardinal_offsets:
        r, c = center_row + dr, center_col + dc
        if (r, c) in grid_positions:
            cardinal_cells.append((r, c))

    if len(cardinal_cells) < 2:
        raise ValueError(
            f"Need at least 2 cardinal neighbors. Grid {grid_spec} gives {len(cardinal_cells)}."
        )

    # Generate pair options: two different cardinal positions with different relations
    pair_options = []
    for c1, c2 in combinations(cardinal_cells, 2):
        # obj1 at c1/c2 (peripheral), obj2 at center (reference)
        # Question: "Where is obj1 in relation to obj2?"
        rel_a = compute_relation(grid_positions[c1], center_pos)
        rel_b = compute_relation(grid_positions[c2], center_pos)
        if rel_a != rel_b:
            pair_options.append((c1, c2, rel_a, rel_b))

    if not pair_options:
        raise ValueError("No valid pair options found with different relations.")

    samples = []
    for pair_id in range(n_pairs):
        c1, c2, rel_a, rel_b = rng.choice(pair_options)

        # Pick objects
        shape1 = rng.choice(SHAPES)  # peripheral (obj1)
        shape2 = rng.choice(SHAPES)  # center (obj2)
        color_names = list(COLORS.keys())
        cn1, cn2 = rng.sample(color_names, 2)

        obj1_name = f"{cn1} {shape1}"
        obj2_name = f"{cn2} {shape2}"

        pos_a = grid_positions[c1]  # obj1 in image A
        pos_b = grid_positions[c2]  # obj1 in image B

        # Image A: obj1 at cardinal position c1, obj2 at center
        img_a = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        draw_a = ImageDraw.Draw(img_a)
        draw_shape(draw_a, shape1, pos_a, shape_size, COLORS[cn1])
        draw_shape(draw_a, shape2, center_pos, shape_size, COLORS[cn2])

        # Image B: obj1 at cardinal position c2, obj2 at center
        img_b = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        draw_b = ImageDraw.Draw(img_b)
        draw_shape(draw_b, shape1, pos_b, shape_size, COLORS[cn1])
        draw_shape(draw_b, shape2, center_pos, shape_size, COLORS[cn2])

        for img, pos, rel in [(img_a, pos_a, rel_a), (img_b, pos_b, rel_b)]:
            samples.append(
                {
                    "image": img,
                    "objects": [obj1_name, obj2_name],
                    "preposition": rel,
                    "pair_id": pair_id,
                    "obj1_position": list(pos),
                    "obj2_position": list(center_pos),
                    "shape_size": shape_size,
                    "image_size": image_size,
                }
            )

    return samples


def generate_paired_dataset(
    n_pairs, image_size, grid_spec, shape_size, seed=42, aligned=True
):
    """Generate paired samples: same objects, swapped positions → flipped relation.

    Each pair shares the same two objects (shape+color) but places them at
    swapped grid cells, so the spatial relation flips (left↔right or above↔below).

    Args:
        n_pairs: Number of pairs to generate.
        image_size: Image width/height in pixels.
        grid_spec: Tuple (rows, cols) for position grid.
        shape_size: Shape diameter in pixels.
        seed: Random seed.
        aligned: If True, only axis-aligned pairs.

    Returns:
        List of sample dicts. Each sample has an extra "pair_id" field.
        Samples at indices 2*i and 2*i+1 form a pair.
    """
    rng = random.Random(seed)
    grid_rows, grid_cols = grid_spec
    grid_positions = get_grid_positions(image_size, grid_rows, grid_cols, shape_size)

    # Find swappable cell pairs (relation flips when positions are swapped)
    swappable = []
    for c1, c2 in combinations(grid_positions.keys(), 2):
        r1, col1 = c1
        r2, col2 = c2

        if aligned and r1 != r2 and col1 != col2:
            continue  # skip diagonals

        rel_a = compute_relation(grid_positions[c1], grid_positions[c2])
        rel_b = compute_relation(grid_positions[c2], grid_positions[c1])

        if rel_a != rel_b:  # relation flips when swapped
            swappable.append((c1, c2, rel_a, rel_b))

    if not swappable:
        raise ValueError(
            f"No swappable pairs found with grid {grid_spec} (aligned={aligned})"
        )

    samples = []
    for pair_id in range(n_pairs):
        c1, c2, rel_a, rel_b = rng.choice(swappable)
        pos1 = grid_positions[c1]
        pos2 = grid_positions[c2]

        # Pick shared objects
        shape1 = rng.choice(SHAPES)
        shape2 = rng.choice(SHAPES)
        color_names = list(COLORS.keys())
        c1_name, c2_name = rng.sample(color_names, 2)

        obj1_name = f"{c1_name} {shape1}"
        obj2_name = f"{c2_name} {shape2}"

        # Image A: obj1 at pos1, obj2 at pos2
        img_a = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        draw_a = ImageDraw.Draw(img_a)
        draw_shape(draw_a, shape1, pos1, shape_size, COLORS[c1_name])
        draw_shape(draw_a, shape2, pos2, shape_size, COLORS[c2_name])

        # Image B: obj1 at pos2, obj2 at pos1 (swapped)
        img_b = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        draw_b = ImageDraw.Draw(img_b)
        draw_shape(draw_b, shape1, pos2, shape_size, COLORS[c1_name])
        draw_shape(draw_b, shape2, pos1, shape_size, COLORS[c2_name])

        samples.append(
            {
                "image": img_a,
                "objects": [obj1_name, obj2_name],
                "preposition": rel_a,
                "pair_id": pair_id,
            }
        )
        samples.append(
            {
                "image": img_b,
                "objects": [obj1_name, obj2_name],
                "preposition": rel_b,
                "pair_id": pair_id,
            }
        )

    return samples


def generate_single_position_pairs(n_pairs, image_size, grid_spec, shape_size, seed=42):
    """Generate paired single-object images for absolute position causal tracing.

    Each pair: same object at two opposite positions (left↔right or above↔below).
    One object per image, clearly on one side.

    Positions use the 4 cardinal cells of the grid (mid-left, mid-right, top-center, bottom-center).
    """
    rng = random.Random(seed)
    grid_rows, grid_cols = grid_spec
    grid_positions = get_grid_positions(image_size, grid_rows, grid_cols, shape_size)

    center_row = grid_rows // 2
    center_col = grid_cols // 2

    # Axis-opposite pairs using cardinal neighbors of grid center.
    # Same positions as the peripheral object in controlled_shapes_pairs (center layout).
    axis_pairs = []
    # Left-right: one step left/right of center
    left_cell = (center_row, center_col - 1)
    right_cell = (center_row, center_col + 1)
    if left_cell in grid_positions and right_cell in grid_positions:
        axis_pairs.append((left_cell, right_cell, "left", "right"))
    # Above-below: one step above/below center
    above_cell = (center_row - 1, center_col)
    below_cell = (center_row + 1, center_col)
    if above_cell in grid_positions and below_cell in grid_positions:
        axis_pairs.append((above_cell, below_cell, "above", "below"))

    if not axis_pairs:
        raise ValueError(f"No axis pairs found with grid {grid_spec}")

    samples = []
    for pair_id in range(n_pairs):
        c1, c2, rel_a, rel_b = rng.choice(axis_pairs)
        pos_a = grid_positions[c1]
        pos_b = grid_positions[c2]

        # Pick one shape+color
        shape = rng.choice(SHAPES_RECOGNITION)
        color_name = rng.choice(list(COLORS.keys()))
        obj_name = f"{color_name} {shape}"

        for pos, rel in [(pos_a, rel_a), (pos_b, rel_b)]:
            img = Image.new("RGB", (image_size, image_size), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw_shape(draw, shape, pos, shape_size, COLORS[color_name])

            samples.append(
                {
                    "image": img,
                    "objects": [obj_name],
                    "preposition": rel,
                    "pair_id": pair_id,
                    "obj1_position": list(pos),
                    "shape_size": shape_size,
                    "image_size": image_size,
                }
            )

    return samples


def generate_single_triplets(n_triplets, image_size, grid_spec, shape_size, seed=42):
    """Generate single-object triplets for localization AND recognition experiments.

    Each triplet has 3 images sharing a triplet_id:
      - anchor:           object X at position P
      - position_pair:    object X at position P' (same appearance, opposite position)
      - recognition_pair: object Y at position P  (different appearance, same position)

    This enables two causal tracing setups from one dataset:
      - Localization: anchor vs position_pair (what changes when position changes?)
      - Recognition:  anchor vs recognition_pair (what changes when object changes?)

    Columns: image, objects, preposition (position label), shape_name,
             triplet_id, variant, obj1_position, shape_size, image_size

    Args:
        n_triplets: Number of triplets (produces 3x images).
        image_size: Image width/height in pixels.
        grid_spec: Tuple (rows, cols) for position grid.
        shape_size: Shape diameter in pixels.
        seed: Random seed.

    Returns:
        List of sample dicts.
    """
    rng = random.Random(seed)
    grid_rows, grid_cols = grid_spec
    grid_positions = get_grid_positions(image_size, grid_rows, grid_cols, shape_size)

    center_row = grid_rows // 2
    center_col = grid_cols // 2

    # Axis-opposite position pairs: left↔right, above↔below
    axis_pairs = []
    left_cell = (center_row, 0)
    right_cell = (center_row, grid_cols - 1)
    if left_cell in grid_positions and right_cell in grid_positions:
        axis_pairs.append((left_cell, right_cell, "left", "right"))
    above_cell = (0, center_col)
    below_cell = (grid_rows - 1, center_col)
    if above_cell in grid_positions and below_cell in grid_positions:
        axis_pairs.append((above_cell, below_cell, "above", "below"))

    if not axis_pairs:
        raise ValueError(f"No axis pairs found with grid {grid_spec}")

    samples = []
    for triplet_id in range(n_triplets):
        c1, c2, rel_a, rel_b = rng.choice(axis_pairs)
        pos_anchor = grid_positions[c1]
        pos_opposite = grid_positions[c2]

        # Anchor object
        shape_anchor = rng.choice(SHAPES_RECOGNITION)
        color_anchor = rng.choice(list(COLORS.keys()))
        obj_anchor = f"{color_anchor} {shape_anchor}"

        # Recognition pair: different shape AND color at same position
        shape_recog = rng.choice([s for s in SHAPES_RECOGNITION if s != shape_anchor])
        color_recog = rng.choice([c for c in COLORS.keys() if c != color_anchor])
        obj_recog = f"{color_recog} {shape_recog}"

        base = {
            "triplet_id": triplet_id,
            "shape_size": shape_size,
            "image_size": image_size,
        }

        # 1. Anchor: object X at position P
        img = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw_shape(draw, shape_anchor, pos_anchor, shape_size, COLORS[color_anchor])
        samples.append({
            **base,
            "image": img,
            "objects": [obj_anchor],
            "preposition": rel_a,
            "shape_name": shape_anchor,
            "variant": "anchor",
            "obj1_position": list(pos_anchor),
        })

        # 2. Position pair: object X at position P' (opposite)
        img = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw_shape(draw, shape_anchor, pos_opposite, shape_size, COLORS[color_anchor])
        samples.append({
            **base,
            "image": img,
            "objects": [obj_anchor],
            "preposition": rel_b,
            "shape_name": shape_anchor,
            "variant": "position_pair",
            "obj1_position": list(pos_opposite),
        })

        # 3. Recognition pair: object Y at position P (same position as anchor)
        img = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw_shape(draw, shape_recog, pos_anchor, shape_size, COLORS[color_recog])
        samples.append({
            **base,
            "image": img,
            "objects": [obj_recog],
            "preposition": rel_a,
            "shape_name": shape_recog,
            "variant": "recognition_pair",
            "obj1_position": list(pos_anchor),
        })

    return samples


def generate_single_recognition_pairs(
    n_pairs, image_size, grid_spec, shape_size, seed=42
):
    """Generate paired single-object images for recognition causal tracing.

    Each pair: two different shapes at the same position, same color.
    The ground truth is the shape name (stored in 'preposition' for pipeline compatibility).
    """
    rng = random.Random(seed)
    grid_rows, grid_cols = grid_spec
    grid_positions = get_grid_positions(image_size, grid_rows, grid_cols, shape_size)

    center_row = grid_rows // 2
    center_col = grid_cols // 2

    # Use cardinal positions (not center — object should be clearly somewhere)
    cardinal_cells = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        cell = (center_row + dr, center_col + dc)
        if cell in grid_positions:
            cardinal_cells.append(cell)

    # Shape pairs (different shapes)
    shape_pairs = []
    for i, s1 in enumerate(SHAPES_RECOGNITION):
        for s2 in SHAPES_RECOGNITION[i + 1 :]:
            shape_pairs.append((s1, s2))

    samples = []
    for pair_id in range(n_pairs):
        cell = rng.choice(cardinal_cells)
        pos = grid_positions[cell]
        s1, s2 = rng.choice(shape_pairs)
        color_name = rng.choice(list(COLORS.keys()))

        for shape in [s1, s2]:
            obj_name = f"{color_name} {shape}"
            img = Image.new("RGB", (image_size, image_size), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw_shape(draw, shape, pos, shape_size, COLORS[color_name])

            samples.append(
                {
                    "image": img,
                    "objects": [obj_name],
                    "preposition": shape,  # GT is the shape name
                    "pair_id": pair_id,
                    "obj1_position": list(pos),
                    "shape_size": shape_size,
                    "image_size": image_size,
                }
            )

    return samples


def generate_attribute_chain_pairs(
    n_pairs, image_size, grid_spec, shape_size, seed=42, task_variant="attribute_chain"
):
    """Generate paired 3-object images for attribute chain and task routing experiments.

    Layout: reference object at center, two peripheral objects of the SAME shape
    but DIFFERENT colors at two cardinal positions. Swapping peripherals changes
    the answer.

    Stores BOTH task variants' metadata in each sample:
        objects_attribute / preposition_attribute: for attribute_chain task
        objects_spatial / preposition_spatial: for spatial_relative task
    Plus legacy objects/preposition for backward compat (uses task_variant to pick).

    Pairs: swap the two peripherals' positions → answer changes for all variants.
    """
    rng = random.Random(seed)
    grid_rows, grid_cols = grid_spec
    grid_positions = get_grid_positions(image_size, grid_rows, grid_cols, shape_size)

    # Center cell
    center_row = grid_rows // 2
    center_col = grid_cols // 2
    center_cell = (center_row, center_col)
    center_pos = grid_positions[center_cell]

    # Cardinal neighbors
    cardinal_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    cardinal_cells = []
    for dr, dc in cardinal_offsets:
        r, c = center_row + dr, center_col + dc
        if (r, c) in grid_positions:
            cardinal_cells.append((r, c))

    if len(cardinal_cells) < 2:
        raise ValueError(
            f"Need at least 2 cardinal neighbors. Grid {grid_spec} gives {len(cardinal_cells)}."
        )

    # Find pairs of cardinal positions with different relations to center
    pair_options = []
    for c1, c2 in combinations(cardinal_cells, 2):
        rel_a = compute_relation(grid_positions[c1], center_pos)
        rel_b = compute_relation(grid_positions[c2], center_pos)
        if rel_a != rel_b:
            pair_options.append((c1, c2, rel_a, rel_b))

    if not pair_options:
        raise ValueError("No valid pair options found with different relations.")

    samples = []
    for pair_id in range(n_pairs):
        c1, c2, rel_a, rel_b = rng.choice(pair_options)

        # Pick target shape (same for both peripherals) and a DIFFERENT reference shape
        target_shape, ref_shape = rng.sample(SHAPES, 2)
        ref_color_name = rng.choice(list(COLORS.keys()))
        color_a, color_b = rng.sample(ANSWER_COLORS, 2)

        ref_name = f"{ref_color_name} {ref_shape}"
        direction = rel_a
        color_a_obj = f"{color_a} {target_shape}"

        pos_1 = grid_positions[c1]
        pos_2 = grid_positions[c2]

        # Image A: color_a at c1, color_b at c2
        img_a = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        draw_a = ImageDraw.Draw(img_a)
        draw_shape(draw_a, target_shape, pos_1, shape_size, COLORS[color_a])
        draw_shape(draw_a, target_shape, pos_2, shape_size, COLORS[color_b])
        draw_shape(draw_a, ref_shape, center_pos, shape_size, COLORS[ref_color_name])

        # Image B: swap peripherals
        img_b = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        draw_b = ImageDraw.Draw(img_b)
        draw_shape(draw_b, target_shape, pos_1, shape_size, COLORS[color_b])
        draw_shape(draw_b, target_shape, pos_2, shape_size, COLORS[color_a])
        draw_shape(draw_b, ref_shape, center_pos, shape_size, COLORS[ref_color_name])

        # Both task variants' metadata
        # Attribute: "What color is the {shape} to the {dir} of the {ref}?"
        objects_attr = [target_shape, direction, ref_name]
        gt_attr_a = color_a
        gt_attr_b = color_b
        # Spatial: "Where is the {color obj} in relation to the {ref}?"
        objects_spat = [color_a_obj, ref_name]
        gt_spat_a = rel_a
        gt_spat_b = rel_b

        # Legacy objects/preposition for backward compat
        if task_variant == "attribute_chain":
            objects = objects_attr
            gt_a, gt_b = gt_attr_a, gt_attr_b
        else:
            objects = objects_spat
            gt_a, gt_b = gt_spat_a, gt_spat_b

        base_sample = {
            "objects": objects,
            "objects_attribute": objects_attr,
            "objects_spatial": objects_spat,
            "obj2_position": list(center_pos),
            "obj3_position": list(pos_2),
            "shape_size": shape_size,
            "image_size": image_size,
        }

        samples.append(
            {
                **base_sample,
                "image": img_a,
                "preposition": gt_a,
                "preposition_attribute": gt_attr_a,
                "preposition_spatial": gt_spat_a,
                "pair_id": pair_id,
                "obj1_position": list(pos_1),
            }
        )
        samples.append(
            {
                **base_sample,
                "image": img_b,
                "preposition": gt_b,
                "preposition_attribute": gt_attr_b,
                "preposition_spatial": gt_spat_b,
                "pair_id": pair_id,
                "obj1_position": (
                    list(pos_1) if task_variant == "attribute_chain" else list(pos_2)
                ),
            }
        )

    return samples


def generate_attribute_shape_pairs(
    n_pairs, image_size, grid_spec, shape_size, seed=42, task_variant="attribute_shape"
):
    """Generate paired 3-object images for shape attribution experiments.

    Layout: reference object at center, two peripheral objects of the SAME color
    but DIFFERENT shapes at two cardinal positions. Swapping peripherals changes
    the answer.

    Stores BOTH task variants' metadata in each sample:
        objects_attribute / preposition_attribute: for attribute_shape task
        objects_spatial / preposition_spatial: for spatial_relative task
    Plus legacy objects/preposition for backward compat (uses task_variant to pick).

    Pairs: swap the two peripherals' positions → answer changes for all variants.
    """
    rng = random.Random(seed)
    grid_rows, grid_cols = grid_spec
    grid_positions = get_grid_positions(image_size, grid_rows, grid_cols, shape_size)

    # Center cell
    center_row = grid_rows // 2
    center_col = grid_cols // 2
    center_cell = (center_row, center_col)
    center_pos = grid_positions[center_cell]

    # Cardinal neighbors
    cardinal_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    cardinal_cells = []
    for dr, dc in cardinal_offsets:
        r, c = center_row + dr, center_col + dc
        if (r, c) in grid_positions:
            cardinal_cells.append((r, c))

    if len(cardinal_cells) < 2:
        raise ValueError(
            f"Need at least 2 cardinal neighbors. Grid {grid_spec} gives {len(cardinal_cells)}."
        )

    # Find pairs of cardinal positions with different relations to center
    pair_options = []
    for c1, c2 in combinations(cardinal_cells, 2):
        rel_a = compute_relation(grid_positions[c1], center_pos)
        rel_b = compute_relation(grid_positions[c2], center_pos)
        if rel_a != rel_b:
            pair_options.append((c1, c2, rel_a, rel_b))

    if not pair_options:
        raise ValueError("No valid pair options found with different relations.")

    samples = []
    for pair_id in range(n_pairs):
        c1, c2, rel_a, rel_b = rng.choice(pair_options)

        # Pick 2 different shapes from answer set for peripherals
        shape_a, shape_b = rng.sample(ANSWER_SHAPES, 2)
        # Pick a DIFFERENT reference shape to avoid collisions
        ref_shape = rng.choice([s for s in SHAPES if s not in (shape_a, shape_b)])
        # Peripheral color (same for both) and a different reference color
        periph_color_name, ref_color_name = rng.sample(list(COLORS.keys()), 2)

        ref_name = f"{ref_color_name} {ref_shape}"
        direction = rel_a
        color_a_obj = f"{periph_color_name} {shape_a}"

        pos_1 = grid_positions[c1]
        pos_2 = grid_positions[c2]

        # Image A: shape_a at c1, shape_b at c2
        img_a = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        draw_a = ImageDraw.Draw(img_a)
        draw_shape(draw_a, shape_a, pos_1, shape_size, COLORS[periph_color_name])
        draw_shape(draw_a, shape_b, pos_2, shape_size, COLORS[periph_color_name])
        draw_shape(draw_a, ref_shape, center_pos, shape_size, COLORS[ref_color_name])

        # Image B: swap peripherals
        img_b = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        draw_b = ImageDraw.Draw(img_b)
        draw_shape(draw_b, shape_a, pos_2, shape_size, COLORS[periph_color_name])
        draw_shape(draw_b, shape_b, pos_1, shape_size, COLORS[periph_color_name])
        draw_shape(draw_b, ref_shape, center_pos, shape_size, COLORS[ref_color_name])

        # Both task variants' metadata
        # Attribute: "What shape is the {color} object to the {dir} of the {ref}?"
        objects_attr = [periph_color_name, direction, ref_name]
        gt_attr_a = shape_a
        gt_attr_b = shape_b
        # Spatial: "Where is the {color obj} in relation to the {ref}?"
        objects_spat = [color_a_obj, ref_name]
        gt_spat_a = rel_a
        gt_spat_b = rel_b

        # Legacy objects/preposition for backward compat
        if task_variant == "attribute_shape":
            objects = objects_attr
            gt_a, gt_b = gt_attr_a, gt_attr_b
        else:
            objects = objects_spat
            gt_a, gt_b = gt_spat_a, gt_spat_b

        base_sample = {
            "objects": objects,
            "objects_attribute": objects_attr,
            "objects_spatial": objects_spat,
            "obj2_position": list(center_pos),
            "obj3_position": list(pos_2),
            "shape_size": shape_size,
            "image_size": image_size,
        }

        samples.append(
            {
                **base_sample,
                "image": img_a,
                "preposition": gt_a,
                "preposition_attribute": gt_attr_a,
                "preposition_spatial": gt_spat_a,
                "pair_id": pair_id,
                "obj1_position": list(pos_1),
            }
        )
        samples.append(
            {
                **base_sample,
                "image": img_b,
                "preposition": gt_b,
                "preposition_attribute": gt_attr_b,
                "preposition_spatial": gt_spat_b,
                "pair_id": pair_id,
                "obj1_position": (
                    list(pos_1) if task_variant == "attribute_shape" else list(pos_2)
                ),
            }
        )

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate controlled shapes dataset for spatial reasoning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of samples to generate (default: 500)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Image size in pixels (default: 1024). "
        "896 -> 28x28=784 tokens (5.2 token gap), "
        "1024 -> 32x32=1024 tokens (6.5 token gap, recommended).",
    )
    parser.add_argument(
        "--shape-size",
        type=int,
        default=80,
        help="Shape diameter in pixels (default: 80)",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="3x3",
        help="Grid layout as ROWSxCOLS (default: 3x3). Shapes placed at grid cell centers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for HuggingFace dataset (default: data/controlled_shapes.hf)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Save N preview images to output_dir/previews/ (default: 0)",
    )
    parser.add_argument(
        "--aligned",
        action="store_true",
        help="Only axis-aligned placements: same row for left/right, same column for above/below.",
    )
    parser.add_argument(
        "--pairs",
        action="store_true",
        help="Generate matched pairs (same objects, swapped positions). "
        "n-samples becomes n-pairs, output has 2x samples with pair_id field.",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Center layout: obj2 (reference) at grid center, obj1 (asked-about) at "
        "cardinal positions. Requires --pairs. Stores object positions for "
        "region-based patch analysis in causal tracing.",
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        choices=["position", "recognition"],
        help="Single-object mode. 'position': same object at opposite positions "
        "(left↔right, above↔below). 'recognition': different shapes at same "
        "position. Both produce paired datasets for causal tracing. "
        "Uses 4 shapes: circle, square, triangle, star.",
    )
    parser.add_argument(
        "--triplet",
        action="store_true",
        help="Single-object triplet mode. Each triplet has 3 images: "
        "anchor (obj X at pos P), position_pair (obj X at pos P'), "
        "recognition_pair (obj Y at pos P). Enables both localization and "
        "recognition causal tracing from one dataset. n-samples = n-triplets, "
        "output has 3x samples with triplet_id and variant fields.",
    )
    parser.add_argument(
        "--attribute-chain",
        action="store_true",
        help="3-object attribute chain mode. Reference at center, two peripherals "
        "of same shape but different colors at cardinal positions. "
        "Question: 'What color is the {shape} to the {dir} of the {ref}?' "
        "Answer is a color. Tests 3-hop: ground ref → locate direction → read color.",
    )
    parser.add_argument(
        "--attribute-shape",
        action="store_true",
        help="3-object attribute shape mode. Reference at center, two peripherals "
        "of same color but different shapes at cardinal positions. "
        "Question: 'What shape is the {color} object to the {dir} of the {ref}?' "
        "Answer is a shape. Tests task routing with shape identification.",
    )
    parser.add_argument(
        "--task-variant",
        type=str,
        default=None,
        choices=[
            "attribute_chain",
            "attribute_shape",
            "spatial_relative",
            "spatial_absolute",
        ],
        help="Task variant for --attribute-chain / --attribute-shape mode. "
        "All variants produce identical images (same seed) but different questions/GTs. "
        "Default: attribute_chain for --attribute-chain, attribute_shape for --attribute-shape. "
        "spatial_relative: 'Where is X in relation to Y?' (GT=direction). "
        "spatial_absolute: 'Where is X in the image?' (GT=direction).",
    )
    args = parser.parse_args()

    # Parse grid spec
    grid_parts = args.grid.split("x")
    if len(grid_parts) != 2:
        parser.error(f"Grid must be ROWSxCOLS, got: {args.grid}")
    grid_spec = (int(grid_parts[0]), int(grid_parts[1]))

    output_path = args.output or "data/controlled_shapes.hf"
    output_path = Path(output_path)

    if args.center and not args.pairs:
        parser.error("--center currently requires --pairs")

    # Default task variant based on mode
    if args.task_variant is None:
        if args.attribute_shape:
            args.task_variant = "attribute_shape"
        else:
            args.task_variant = "attribute_chain"

    if args.attribute_shape:
        mode_str = "attribute-shape pairs"
    elif args.attribute_chain:
        mode_str = "attribute-chain pairs"
    elif args.triplet:
        mode_str = "single-object triplets"
    elif args.single:
        mode_str = f"single-object {args.single} pairs"
    elif args.center:
        mode_str = "center pairs"
    elif args.pairs:
        mode_str = "pairs"
    else:
        mode_str = "samples"

    shapes_used = SHAPES_RECOGNITION if (args.single or args.triplet) else SHAPES
    print(f"Generating {args.n_samples} {mode_str}...")
    print(f"  Image size: {args.image_size}x{args.image_size}")
    print(f"  Shape size: {args.shape_size}px")
    print(
        f"  Grid: {grid_spec[0]}x{grid_spec[1]} ({grid_spec[0] * grid_spec[1]} positions)"
    )
    print(f"  Aligned: {args.aligned}")
    print(f"  Paired: {args.pairs}")
    print(f"  Center: {args.center}")
    print(f"  Single: {args.single}")
    if args.single == "recognition":
        print(f"  Alternatives: {sorted(SHAPES_RECOGNITION)}")
    elif args.attribute_chain:
        print(f"  Answer colors: {sorted(ANSWER_COLORS)}")
    elif args.attribute_shape:
        print(f"  Answer shapes: {sorted(ANSWER_SHAPES)}")
    else:
        print(f"  Prepositions: {sorted(PREPOSITIONS)}")
    print(f"  Shapes: {shapes_used}")
    print(f"  Colors: {sorted(COLORS.keys())}")

    if args.attribute_shape:
        samples = generate_attribute_shape_pairs(
            n_pairs=args.n_samples,
            image_size=args.image_size,
            grid_spec=grid_spec,
            shape_size=args.shape_size,
            seed=args.seed,
            task_variant=args.task_variant,
        )
    elif args.attribute_chain:
        samples = generate_attribute_chain_pairs(
            n_pairs=args.n_samples,
            image_size=args.image_size,
            grid_spec=grid_spec,
            shape_size=args.shape_size,
            seed=args.seed,
            task_variant=args.task_variant,
        )
    elif args.triplet:
        samples = generate_single_triplets(
            n_triplets=args.n_samples,
            image_size=args.image_size,
            grid_spec=grid_spec,
            shape_size=args.shape_size,
            seed=args.seed,
        )
    elif args.single == "position":
        samples = generate_single_position_pairs(
            n_pairs=args.n_samples,
            image_size=args.image_size,
            grid_spec=grid_spec,
            shape_size=args.shape_size,
            seed=args.seed,
        )
    elif args.single == "recognition":
        samples = generate_single_recognition_pairs(
            n_pairs=args.n_samples,
            image_size=args.image_size,
            grid_spec=grid_spec,
            shape_size=args.shape_size,
            seed=args.seed,
        )
    elif args.center:
        samples = generate_center_paired_dataset(
            n_pairs=args.n_samples,
            image_size=args.image_size,
            grid_spec=grid_spec,
            shape_size=args.shape_size,
            seed=args.seed,
        )
    elif args.pairs:
        samples = generate_paired_dataset(
            n_pairs=args.n_samples,
            image_size=args.image_size,
            grid_spec=grid_spec,
            shape_size=args.shape_size,
            seed=args.seed,
            aligned=args.aligned,
        )
    else:
        samples = generate_balanced_dataset(
            n_samples=args.n_samples,
            image_size=args.image_size,
            grid_spec=grid_spec,
            shape_size=args.shape_size,
            seed=args.seed,
            aligned=args.aligned,
        )

    # Print distribution
    from collections import Counter

    prep_counts = Counter(s["preposition"] for s in samples)
    print(f"\nGenerated {len(samples)} samples:")
    for p in sorted(prep_counts):
        print(f"  {p}: {prep_counts[p]}")

    # Save preview images
    if args.preview > 0:
        preview_dir = output_path.parent / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for i, sample in enumerate(samples[: args.preview]):
            objs = sample["objects"]
            rel = sample["preposition"]
            obj_str = "_".join(o.replace(" ", "_") for o in objs)
            fname = f"{i:03d}_{obj_str}_{rel}.png"
            sample["image"].save(preview_dir / fname)
            print(f"  Saved preview: {fname}")
        print(f"Previews saved to: {preview_dir}")

    # Convert to HuggingFace dataset
    data_dict = {
        "image": [s["image"] for s in samples],
        "objects": [s["objects"] for s in samples],
        "preposition": [s["preposition"] for s in samples],
    }
    if args.triplet:
        data_dict["triplet_id"] = [s["triplet_id"] for s in samples]
        data_dict["variant"] = [s["variant"] for s in samples]
        data_dict["shape_name"] = [s["shape_name"] for s in samples]
        data_dict["obj1_position"] = [s["obj1_position"] for s in samples]
        data_dict["shape_size"] = [s["shape_size"] for s in samples]
        data_dict["image_size"] = [s["image_size"] for s in samples]
    if (
        args.pairs
        or args.center
        or args.single
        or args.attribute_chain
        or args.attribute_shape
    ):
        data_dict["pair_id"] = [s["pair_id"] for s in samples]
    if args.center or args.single or args.attribute_chain or args.attribute_shape:
        data_dict["obj1_position"] = [s["obj1_position"] for s in samples]
        data_dict["shape_size"] = [s["shape_size"] for s in samples]
        data_dict["image_size"] = [s["image_size"] for s in samples]
    if args.center or args.attribute_chain or args.attribute_shape:
        data_dict["obj2_position"] = [s["obj2_position"] for s in samples]
    if args.attribute_chain or args.attribute_shape:
        data_dict["obj3_position"] = [s["obj3_position"] for s in samples]
        data_dict["objects_attribute"] = [s["objects_attribute"] for s in samples]
        data_dict["objects_spatial"] = [s["objects_spatial"] for s in samples]
        data_dict["preposition_attribute"] = [
            s["preposition_attribute"] for s in samples
        ]
        data_dict["preposition_spatial"] = [s["preposition_spatial"] for s in samples]
    dataset = Dataset.from_dict(data_dict)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    print(f"\nDataset saved to: {output_path}")
    print(f"Load with: dataset = load_from_disk('{output_path}')")


if __name__ == "__main__":
    main()
