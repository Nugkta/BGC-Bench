"""Command-line interface: ``bgcbench run | embed | evaluate | list``."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path


def _encoder(p: argparse.ArgumentParser) -> None:
    p.add_argument("--encoder", required=True, metavar="NAME",
                   help="built-in encoder (see bgcbench list)")
    p.add_argument("--device", default=None, metavar="DEVICE",
                   help="embedding device (default: cuda when visible, else cpu; encoders that "
                        "embed on a GPU stop instead of using the CPU unless given --device "
                        "cpu, and LC-PLM and Evo 2 need CUDA)")
    p.add_argument("--refresh", action="store_true", help="recompute cached embeddings")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tasks", nargs="+", default=None, metavar="TASK",
                   help="tasks to run (default: the five official tasks)")
    p.add_argument("--task-module", action="append", default=[], metavar="PATH_OR_MODULE",
                   help="import a .py file or module that registers extra tasks (repeatable)")
    p.add_argument("--chem-encoder", default=None, metavar="NAME",
                   help="retrieval with a shipped molecule vector set instead of MoLFormer "
                        "(tasks/chem_encoders/encoders.json, e.g. ecfp4)")
    p.add_argument("--out", type=Path, default=None, metavar="PATH",
                   help="result JSON (default: results/<name>.json)")
    p.add_argument("--n-boot", type=int, default=500, metavar="N",
                   help="bootstrap resamples (default: 500)")
    p.add_argument("--threads", type=int, default=None, metavar="N",
                   help="override the per-task protocol CPU thread counts")
    p.add_argument("--retrieval-device", default="cpu", metavar="DEVICE",
                   help="device passed to the tasks; the retrieval head trains on it "
                        "(default: cpu)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="bgcbench",
        description="Embed BGCs with a built-in encoder and score embeddings on BGC-Bench.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="embed the dataset with a built-in encoder and evaluate")
    _encoder(p_run)
    _common(p_run)

    p_embed = sub.add_parser("embed", help="compute and cache embeddings without scoring")
    _encoder(p_embed)

    p_ev = sub.add_parser("evaluate", help="evaluate your own embeddings")
    p_ev.add_argument("--embeddings", required=True, type=Path, metavar="PATH",
                      help=".npz with arrays bgc_ids and embeddings")
    p_ev.add_argument("--adomain-embeddings", type=Path, default=None, metavar="PATH",
                      help=".npz with domain_ids and embeddings (substrate task); "
                           "same as --rep adomain=PATH")
    p_ev.add_argument("--chem-embeddings", type=Path, default=None, metavar="PATH",
                      help=".npz with compound_ids and embeddings (replaces MoLFormer in "
                           "retrieval); same as --rep molecule=PATH")
    p_ev.add_argument("--rep", action="append", default=[], metavar="KIND=PATH",
                      help=".npz of another representation kind (repeatable)")
    p_ev.add_argument("--name", default=None, help="name for the result (default: file stem)")
    _common(p_ev)

    sub.add_parser("list", help="list the built-in encoders")
    args = ap.parse_args(argv)

    import bgcbench
    from .encoders.base import REGISTRY, NoGPUError, encoder_class
    from .tasks.base import select

    if args.cmd == "list":
        print(f"{'name':<13}{'model':<25}{'input':<12}{'device':<8}{'uv env':<10}licence")
        for name, spec in REGISTRY.items():
            cls = encoder_class(name)
            device = "cuda" if cls.cuda_only else "gpu" if cls.gpu else "cpu"
            env = "envs/gpu" if spec.env == "gpu" else "main"
            print(f"{name:<13}{spec.display:<25}{spec.input:<12}{device:<8}{env:<10}"
                  f"{spec.licence}")
        return 0

    if getattr(args, "encoder", None) is not None and args.encoder not in REGISTRY:
        ap.error(f"unknown encoder {args.encoder!r}; built-in: {', '.join(REGISTRY)}")
    if args.cmd == "embed":
        try:
            bgc, ad = bgcbench.embed(args.encoder, device=args.device, refresh=args.refresh)
        except (NoGPUError, ImportError) as exc:
            ap.exit(1, f"bgcbench: error: {exc}\n")
        n_ad = 0 if ad is None else len(ad.ids)
        print(f"[bgcbench] cached {len(bgc.ids)} BGC and {n_ad} A-domain vectors")
        return 0

    for spec in args.task_module:
        _import(spec)
    try:
        select(args.tasks)
    except ValueError as exc:
        ap.error(str(exc))
    chem = getattr(args, "chem_embeddings", None)
    if args.chem_encoder:
        if chem is not None:
            ap.error("give --chem-encoder or --chem-embeddings, not both")
        from .tasks.retrieval import chem_encoder_path

        try:
            chem = chem_encoder_path(args.chem_encoder)
        except ValueError as exc:
            ap.error(str(exc))
    opts = dict(tasks=args.tasks, n_boot=args.n_boot, threads=args.threads,
                retrieval_device=args.retrieval_device)
    if args.cmd == "run":
        print(f"[bgcbench] {args.encoder}: embedding (cached after the first run) ...")
        try:
            result = bgcbench.run(args.encoder, device=args.device, refresh=args.refresh,
                                  chem_embeddings=chem, **opts)
        except (NoGPUError, ImportError) as exc:
            ap.exit(1, f"bgcbench: error: {exc}\n")
    else:
        reps = {}
        for item in args.rep:
            kind, sep, path = item.partition("=")
            if not (kind and sep and path):
                ap.error(f"--rep expects KIND=PATH, got {item!r}")
            reps[kind] = Path(path)
        chem_flag = "--chem-encoder" if args.chem_encoder else "--chem-embeddings"
        for kind, flag in (("adomain", "--adomain-embeddings"), ("molecule", chem_flag)):
            if kind in reps and (chem if kind == "molecule" else args.adomain_embeddings):
                ap.error(f"give {flag} or --rep {kind}=PATH, not both")
        name = args.name or args.embeddings.stem
        result = bgcbench.evaluate(args.embeddings, args.adomain_embeddings, chem, reps=reps,
                                   name=name, **opts)
    out = args.out or Path("results") / f"{result.name}.json"
    result.to_json(out)
    print()
    print(result.table())
    print(f"\n[bgcbench] wrote {out}")
    return 0


def _import(spec: str) -> None:
    """Import a task module, given as a ``.py`` path or a dotted module name."""
    path = Path(spec)
    if path.suffix != ".py":
        importlib.import_module(spec)
        return
    if not path.is_file():
        raise FileNotFoundError(f"task module {spec} not found")
    mod_spec = importlib.util.spec_from_file_location(f"bgcbench_tasks_{path.stem}", path)
    module = importlib.util.module_from_spec(mod_spec)
    sys.modules[mod_spec.name] = module
    mod_spec.loader.exec_module(module)


if __name__ == "__main__":
    sys.exit(main())
