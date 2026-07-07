from pathlib import Path

from datasets import load_from_disk


def load_dataset(dataset_path, dataset_name, task_type=None, perspective=None):
    """
    Load HuggingFace dataset and extract prepositions.

    Args:
        dataset_path: Base path to datasets directory
        dataset_name: Name of the dataset (e.g., "controlled_clevr")
        task_type: Optional task type to select columns from multi-task datasets.
            For datasets with suffixed columns (objects_spatial, objects_attribute, etc.),
            maps them to standard "objects" and "preposition" columns.
        perspective: Optional perspective for COMFORT datasets.
            - None or "camera": use "preposition" column (camera POV, default)
            - "addressee": use "preposition_sophia" column as GT
            - "main": use "preposition_main" column as GT

    Returns:
        Tuple of (dataset, prepositions_set)
    """
    full_path = Path(dataset_path) / f"{dataset_name}.hf"
    print(f"Loading dataset from {full_path}...")
    dataset = load_from_disk(str(full_path))

    # Multi-task datasets store suffixed columns; remap based on task_type
    # Drop legacy objects/preposition first, then rename the selected variant
    if task_type in ("attribute_chain", "attribute_shape") and "objects_attribute" in dataset.column_names:
        dataset = dataset.remove_columns(["objects", "preposition"])
        dataset = dataset.rename_columns({
            "objects_attribute": "objects",
            "preposition_attribute": "preposition",
        })
        for col in ["objects_spatial", "preposition_spatial"]:
            if col in dataset.column_names:
                dataset = dataset.remove_columns(col)
    elif task_type == "spatial_relative" and "objects_spatial" in dataset.column_names:
        dataset = dataset.remove_columns(["objects", "preposition"])
        dataset = dataset.rename_columns({
            "objects_spatial": "objects",
            "preposition_spatial": "preposition",
        })
        for col in ["objects_attribute", "preposition_attribute"]:
            if col in dataset.column_names:
                dataset = dataset.remove_columns(col)

    # COMFORT perspective remapping: swap GT column based on perspective
    PERSPECTIVE_GT_COLUMN = {
        "addressee": "preposition_sophia",
        "main": "preposition_main",
    }
    gt_col = PERSPECTIVE_GT_COLUMN.get(perspective)
    if gt_col is not None:
        if gt_col not in dataset.column_names:
            raise ValueError(
                f"Perspective '{perspective}' requires column '{gt_col}' "
                f"but dataset only has: {dataset.column_names}"
            )
        dataset = dataset.remove_columns(["preposition"])
        dataset = dataset.rename_column(gt_col, "preposition")

    prepositions = set(dataset["preposition"])
    print(f"Dataset loaded: {len(dataset)} samples, prepositions: {prepositions}")

    return dataset, prepositions


def create_question(objects, prepositions, question_format="forced_choice",
                    task_type="spatial_relative", perspective=None):
    """
    Create spatial reasoning question from objects and prepositions.

    Args:
        objects: List of object names [object1, object2] or [object1] for single-object tasks
        prepositions: Set or list of answer alternatives (prepositions, positions, or shape names)
        question_format: One of:
            - "forced_choice": includes all answer options (default, current behavior)
            - "open": asks for a single word, no options listed
            - "free": bare question, no answer instructions
        task_type: One of:
            - "spatial_relative": two objects, relational (default)
            - "spatial_absolute": single object, absolute position in image
            - "recognition": what object is in the image
        perspective: Optional perspective prefix for POV questions.
            - None: no prefix (default)
            - "camera": "From the camera's viewpoint, ..."
            - "addressee": "From the woman's viewpoint, ..."
            - Any string: "From the {perspective}'s viewpoint, ..."

    Returns:
        Question string
    """
    if task_type == "spatial_relative":
        object1, object2 = objects[0], objects[1]
        base = f"Where is the {object1} in relation to the {object2}?"
    elif task_type == "spatial_absolute":
        object1 = objects[0]
        base = f"Where is the {object1} in the image?"
    elif task_type == "recognition":
        # Shapes recognition. COCO recognition does not use this template; it
        # passes a per-sample question_text (e.g. "What is the person holding?").
        base = "What is the shape of the object in the image?"
    elif task_type == "attribute_chain":
        target_shape = objects[0]   # e.g., "circle"
        direction = objects[1]       # e.g., "left"
        reference = objects[2]       # e.g., "blue cube"
        base = f"What color is the {target_shape} {direction} of the {reference}?"
    elif task_type == "attribute_shape":
        target_color = objects[0]    # e.g., "red"
        direction = objects[1]       # e.g., "left"
        reference = objects[2]       # e.g., "blue square"
        base = f"What shape is the {target_color} object {direction} of the {reference}?"
    elif task_type == "spatial_or":
        # Alternative relational template with candidate prepositions inline
        # *before* obj2. Used to test whether the prior-work "obj2 enrichment"
        # claim is a positional artifact of preps living downstream of obj2.
        object1, object2 = objects[0], objects[1]
        prep_list = sorted(list(prepositions))
        if len(prep_list) == 1:
            prep_phrase = prep_list[0]
        elif len(prep_list) == 2:
            prep_phrase = f"{prep_list[0]} or {prep_list[1]}"
        else:
            prep_phrase = ", ".join(prep_list[:-1]) + f", or {prep_list[-1]}"
        base = f"Is the {object1} to the {prep_phrase} of the {object2}?"
    else:
        raise ValueError(
            f"Unknown task_type: {task_type!r}. "
            f"Expected 'spatial_relative', 'spatial_absolute', 'recognition', "
            f"'attribute_chain', 'attribute_shape', or 'spatial_or'."
        )

    # Add perspective prefix
    if perspective is not None:
        PERSPECTIVE_LABELS = {
            "camera": "the camera",
            "addressee": "the woman",
        }
        if perspective == "main":
            # Main object = reference object (obj2)
            label = f"the {objects[1]}"
        else:
            label = PERSPECTIVE_LABELS.get(perspective, perspective)
        # Lowercase the first letter of base since it follows the prefix
        base = f"From {label}'s viewpoint, {base[0].lower()}{base[1:]}"

    if question_format == "forced_choice":
        alt_list = sorted(list(prepositions))
        alt_options = ", ".join(alt_list)
        return f"{base} Answer only with {alt_options}."
    elif question_format == "open":
        return f"{base} Answer with a single word."
    elif question_format == "free":
        return base
    else:
        raise ValueError(
            f"Unknown question_format: {question_format!r}. "
            f"Expected 'forced_choice', 'open', or 'free'."
        )


def find_token_ranges(input_ids, tokenizer):
    """
    Find token ranges for different parts of the input sequence.

    Identifies the boundaries of:
    - prompt_start: tokens before image
    - image: image tokens (between vision_start and vision_end)
    - text: text tokens (after image, before im_end)
    - prompt_end: tokens after text
    - last: last token position (-1)

    Args:
        input_ids: Token IDs tensor [batch, seq_len] or [seq_len]
        tokenizer: Tokenizer for decoding tokens

    Returns:
        Dict with keys: "prompt_start", "image", "text", "prompt_end", "last"
        Each value is a tuple (start, end) for ranges or int for last
    """
    if input_ids.dim() == 2:
        input_ids = input_ids[0]

    tokens = [tokenizer.decode([t]) for t in input_ids]
    ids = input_ids.tolist()

    # --- Detect image token style ---
    # Qwen3-VL: <|vision_start|> ... image_tokens ... <|vision_end|>
    # InternVL: <img> ... image_tokens ... </img> (ChatML delimiters)
    # LLaVA-OneVision: repeated <image> tokens (token_id 151646)
    vision_start = None
    vision_end = None
    text_end = None

    for i, token in enumerate(tokens):
        if "<|vision_start|>" in token:
            vision_start = i
        if "<|vision_end|>" in token:
            vision_end = i
        # InternVL: <img> / </img> markers
        if token.strip() == "<img>" and vision_start is None:
            vision_start = i
        if token.strip() == "</img>" and vision_end is None:
            vision_end = i
        if "<|im_end|>" in token and vision_end is not None and i > vision_end:
            text_end = i
            break

    # LLaVA fallback: find contiguous block of <image> tokens
    if vision_start is None or vision_end is None:
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        if isinstance(image_token_id, int) and image_token_id in ids:
            first_img = ids.index(image_token_id)
            last_img = first_img
            for j in range(first_img, len(ids)):
                if ids[j] == image_token_id:
                    last_img = j
                else:
                    break

            # LLaVA has no delimiter tokens — compute ranges directly
            img_range = (first_img, last_img + 1)  # half-open: all <image> tokens
            txt_start = last_img + 1                # first token after images

            # Find text_end: first end-of-turn marker after image block
            # - LLaVA-OV (ChatML): <|im_end|>
            # - LLaVA-1.5 (Vicuna): ASSISTANT:
            # - LLaVA-1.6 (Mistral): [/INST]
            txt_end = len(tokens)
            for i in range(txt_start, len(tokens)):
                tok = tokens[i]
                if "<|im_end|>" in tok:
                    txt_end = i
                    break
                if "[/INST]" in tok:
                    txt_end = i
                    break

            # Vicuna: "ASSISTANT" may be split into subword tokens (e.g.
            # ["▁ASS","IST","ANT"] in LLaMA tokenizer), so match on token IDs
            # rather than decoded strings.
            if txt_end == len(tokens):
                assistant_ids = tokenizer.encode(
                    "ASSISTANT", add_special_tokens=False
                )
                for i in range(txt_start, len(ids) - len(assistant_ids) + 1):
                    if ids[i : i + len(assistant_ids)] == assistant_ids:
                        txt_end = i
                        break

            if txt_end == len(tokens):
                import warnings
                warnings.warn(
                    "Could not find end-of-turn marker for LLaVA input. "
                    "Text range may include prompt/generation tokens. "
                    f"Last 10 tokens: {tokens[-10:]}"
                )

            return {
                "prompt_start": (0, first_img),
                "image": img_range,
                "text": (txt_start, txt_end),
                "prompt_end": (txt_end, len(tokens) - 1),
                "last": -1,
            }

    if vision_start is None or vision_end is None:
        # No image tokens (text-only input)
        im_start_positions = [i for i, t in enumerate(tokens) if "<|im_start|>" in t]
        im_end_positions = [i for i, t in enumerate(tokens) if "<|im_end|>" in t]

        if len(im_start_positions) < 2 or len(im_end_positions) < 2:
            raise ValueError(
                f"Could not find user message boundaries in text-only input. Tokens: {tokens[:30]}..."
            )

        text_start = im_start_positions[1] + 1
        text_end = im_end_positions[1]
        while text_start < text_end:
            t = tokens[text_start]
            if "<|" not in t and t.strip().lower() not in ("", "user"):
                break
            text_start += 1

        return {
            "prompt_start": (0, text_start),
            "image": None,
            "text": (text_start, text_end),
            "prompt_end": (text_end, len(tokens) - 1),
            "last": -1,
        }

    prompt_start_range = (0, vision_start)
    image_range = (vision_start + 1, vision_end)
    text_start = vision_end + 1
    if text_end is None:
        text_end = len(tokens)
    text_range = (text_start, text_end)
    prompt_end_range = (text_end, len(tokens) - 1)
    last_token = -1

    return {
        "prompt_start": prompt_start_range,
        "image": image_range,
        "text": text_range,
        "prompt_end": prompt_end_range,
        "last": last_token,
    }


def find_cluster_ranges(input_ids, tokenizer, objects, ground_truth=None, prepositions=None):
    """
    Find token ranges for semantic clusters in the question.

    For question: "Where is the {obj1} in relation to the {obj2}? Answer with ..."

    Clusters:
    - object1: tokens for obj1 (single slice)
    - object2: tokens for obj2 (single slice)
    - relation: "Where is the" + "in relation to the" (list of slices, non-contiguous)
    - format: "? Answer with ..." tokens (single slice)

    Args:
        input_ids: Token IDs tensor [batch, seq_len] or [seq_len]
        tokenizer: Tokenizer for decoding tokens
        objects: List [obj1, obj2] object names

    Returns:
        Dict with cluster name -> (start, end) tuple or list of (start, end) tuples
    """
    if input_ids.dim() == 2:
        input_ids = input_ids[0]

    # First get the text range (after image tokens)
    base_ranges = find_token_ranges(input_ids, tokenizer)
    text_start, text_end = base_ranges["text"]

    # Decode text portion token by token
    text_tokens = []
    for i in range(text_start, text_end):
        tok = tokenizer.decode([input_ids[i]])
        text_tokens.append((i, tok))

    # Build full text string with position mapping
    full_text = ""
    char_to_token = []  # char_idx -> token_idx
    for tok_idx, tok_str in text_tokens:
        for _ in tok_str:
            char_to_token.append(tok_idx)
        full_text += tok_str

    def find_substring_span(substring, after=0):
        """Find (start_token, end_token) span for a substring."""
        start_char = full_text.find(substring, after)
        if start_char == -1:
            return None
        end_char = start_char + len(substring)

        # Get token indices covering this span
        if start_char >= len(char_to_token) or end_char > len(char_to_token):
            return None

        start_tok = char_to_token[start_char]
        end_tok = char_to_token[end_char - 1] + 1  # exclusive end
        return (start_tok, end_tok)

    obj1 = objects[0]
    obj2 = objects[1] if len(objects) > 1 else None

    # Detect attribute tasks: "What color/shape is ... to the {dir} of the {ref}?"
    # objects = [target, direction, reference] for attribute_chain / attribute_shape
    what_color_span = find_substring_span("What color is")
    what_shape_span = find_substring_span("What shape is")
    attribute_span = what_color_span or what_shape_span
    if attribute_span is not None and len(objects) == 3:
        # Attribute chain/shape task
        target, direction, reference = objects
        if what_color_span is not None:
            obj1_span = find_substring_span(f"the {target}")
        else:
            obj1_span = find_substring_span(f"the {target} object")
        obj2_span = find_substring_span(f"the {reference}")
        where_span = attribute_span
        context_span = find_substring_span(f"{direction} of")
        relation_spans = [s for s in [where_span, context_span] if s is not None]
    else:
        # Standard tasks
        # Find each cluster (include "the" with objects, exclude from relation)
        obj1_span = find_substring_span(f"the {obj1}")
        obj2_span = find_substring_span(f"the {obj2}") if obj2 else None

        # Relation components (non-contiguous: "Where is" + "in relation to")
        where_span = find_substring_span("Where is")
        if where_span is None:
            # Recognition task: "What is in the image?"
            where_span = find_substring_span("What is")
        in_relation_span = find_substring_span("in relation to")
        in_the_image_span = find_substring_span("in the image")
        # Use whichever context phrase is present
        context_span = in_relation_span or in_the_image_span
        relation_spans = [s for s in [where_span, context_span] if s is not None]

    # Format components
    question_mark_span = find_substring_span("?")
    # Support both "Answer with" and "Answer only with" formats
    answer_with_span = find_substring_span("Answer only with")
    answer_keyword = "Answer only with"
    if answer_with_span is None:
        answer_with_span = find_substring_span("Answer with")
        answer_keyword = "Answer with"
    format_span = find_substring_span(f"? {answer_keyword}")

    # Preposition and punctuation clusters (need answer section position)
    correct_span = None
    wrong_spans = []
    punct_spans = []
    answer_pos = full_text.find(answer_keyword)

    if ground_truth is not None and prepositions is not None and answer_pos >= 0:
        search_after = answer_pos + len(answer_keyword)
        correct_span = find_substring_span(ground_truth, after=search_after)
        for p in sorted(prepositions):
            if p != ground_truth:
                span = find_substring_span(p, after=search_after)
                if span is not None:
                    wrong_spans.append(span)

    # Preposition clusters in the question BODY (preps appearing before the
    # answer keyword, used by templates like spatial_or where candidate
    # prepositions are inline in the question rather than in the answer suffix).
    body_correct_span = None
    body_wrong_spans = []
    if prepositions is not None:
        body_end = answer_pos if answer_pos >= 0 else len(full_text)
        for p in sorted(prepositions):
            start_char = full_text.find(p)
            if start_char < 0:
                continue
            end_char = start_char + len(p)
            if end_char > body_end:
                continue
            if start_char >= len(char_to_token) or end_char > len(char_to_token):
                continue
            start_tok = char_to_token[start_char]
            end_tok = char_to_token[end_char - 1] + 1
            span = (start_tok, end_tok)
            if ground_truth is not None and p == ground_truth:
                body_correct_span = span
            else:
                body_wrong_spans.append(span)

    # Spatial-relator phrase in the question body (e.g., "to the" and "of"
    # bracketing the prep list in the spatial_or template). Analogous to
    # "in relation to" in the standard relational template — captures the
    # connective glue that holds the relational structure together.
    body_relator_spans = []
    body_end_char = answer_pos if answer_pos >= 0 else len(full_text)
    for phrase in ["to the", "of"]:
        start_char = full_text.find(phrase)
        if start_char < 0:
            continue
        end_char = start_char + len(phrase)
        if end_char > body_end_char:
            continue
        if start_char >= len(char_to_token) or end_char > len(char_to_token):
            continue
        start_tok = char_to_token[start_char]
        end_tok = char_to_token[end_char - 1] + 1
        body_relator_spans.append((start_tok, end_tok))

    # Punctuation in answer section: commas, " or ", "."
    if answer_pos >= 0:
        search_after = answer_pos + len(answer_keyword)
        pos = search_after
        while True:
            idx = full_text.find(",", pos)
            if idx == -1:
                break
            span = find_substring_span(",", after=pos)
            if span:
                punct_spans.append(span)
            pos = idx + 1
        or_span = find_substring_span(" or ", after=search_after)
        if or_span:
            punct_spans.append(or_span)
        period_span = find_substring_span(".", after=search_after)
        if period_span:
            punct_spans.append(period_span)

    return {
        # Merged clusters
        "object1": obj1_span,
        "object2": obj2_span,
        "relation": relation_spans,
        "format": format_span,
        # Fine-grained
        "where": where_span,
        "in_relation": context_span,
        "question_mark": question_mark_span,
        "answer_with": answer_with_span,
        "correct_prep": correct_span,
        "wrong_preps": wrong_spans if wrong_spans else None,
        "body_correct_prep": body_correct_span,
        "body_wrong_preps": body_wrong_spans if body_wrong_spans else None,
        "body_relator": body_relator_spans if body_relator_spans else None,
        "punctuation": punct_spans if punct_spans else None,
    }
