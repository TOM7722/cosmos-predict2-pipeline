#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def collect_pairs(root: Path, text_subdir: str, image_subdir: str, limit: int | None, domain_name: str):
    text_root = root / text_subdir
    image_root = root / image_subdir

    if not text_root.is_dir():
        raise SystemExit(f"Expected directory does not exist: {text_root}")
    if not image_root.is_dir():
        raise SystemExit(f"Expected directory does not exist: {image_root}")

    text_files = {
        path.relative_to(text_root).with_suffix("").as_posix(): path
        for path in sorted(text_root.rglob("*.txt"))
    }
    image_files = {
        path.relative_to(image_root).with_suffix("").as_posix(): path
        for path in sorted(image_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }

    missing_images = sorted(set(text_files) - set(image_files))
    missing_texts = sorted(set(image_files) - set(text_files))
    if missing_images:
        raise SystemExit(f"{domain_name}: missing matching images for: {missing_images[:10]}")
    if missing_texts:
        raise SystemExit(f"{domain_name}: missing matching text files for: {missing_texts[:10]}")

    keys = sorted(text_files)
    if limit is not None:
        keys = keys[:limit]

    pairs = []
    for key in keys:
        prompt = text_files[key].read_text(encoding="utf-8").strip()
        if not prompt:
            raise SystemExit(f"{domain_name}: empty prompt file: {text_files[key]}")
        pairs.append((key, prompt, image_files[key]))
    return pairs


def sanitize_name(name: str) -> str:
    cleaned = []
    for ch in name:
        if ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ego-root", default=None,
                        help="Root directory for ego-view data. Omit or set --ego-limit 0 to skip.")
    parser.add_argument("--open-domain-root", default=None,
                        help="Root directory for open-domain data. Omit or set --open-limit 0 to skip.")
    parser.add_argument("--text-subdir", default="caption")
    parser.add_argument("--image-subdir", default="imgs")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--seeds-per-sample", type=int, default=10)
    parser.add_argument("--frames-per-video", type=int, default=160)
    parser.add_argument("--chunk-overlap", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--ego-limit", type=int, default=None)
    parser.add_argument("--open-limit", type=int, default=None)
    args = parser.parse_args()

    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    ego_pairs = []
    if args.ego_root and args.ego_limit != 0:
        ego_root = Path(args.ego_root).expanduser().resolve()
        ego_pairs = collect_pairs(ego_root, args.text_subdir, args.image_subdir, args.ego_limit, "ego")

    open_pairs = []
    if args.open_domain_root and args.open_limit != 0:
        open_root = Path(args.open_domain_root).expanduser().resolve()
        open_pairs = collect_pairs(open_root, args.text_subdir, args.image_subdir, args.open_limit, "open_domain")

    if not ego_pairs and not open_pairs:
        raise SystemExit("No samples to process. Provide --ego-root or --open-domain-root with a non-zero limit.")

    entries = []
    seen_names = set()
    for domain_name, pairs in (("ego", ego_pairs), ("open_domain", open_pairs)):
        for key, prompt, image_path in pairs:
            base = sanitize_name(key)
            for seed in range(args.seeds_per_sample):
                sample_name = f"{domain_name}_{base}_seed{seed:03d}"
                if sample_name in seen_names:
                    raise SystemExit(f"Duplicate sample name generated: {sample_name}")
                seen_names.add(sample_name)
                entries.append(
                    {
                        "name": sample_name,
                        "inference_type": "image2world",
                        "prompt": prompt,
                        "input_path": str(image_path),
                        "seed": seed,
                        "num_output_frames": args.frames_per_video,
                        "enable_autoregressive": True,
                        "chunk_size": 77,
                        "chunk_overlap": args.chunk_overlap,
                        "num_steps": args.num_steps,
                    }
                )

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    print(f"Wrote {len(entries)} manifest rows to {output_jsonl}")
    print(f"Matched pairs: ego={len(ego_pairs)}, open_domain={len(open_pairs)}")
    print(f"Seeds per pair: {args.seeds_per_sample}")
    print(f"Frames per video: {args.frames_per_video}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
