"""Lab 5 training entry point: one arm, one seed, one run, one results record.

Why a script and not notebook cells
----------------------------------
In 2025/26 the training code lived only in notebooks, so nothing was persisted
(``grep`` for ``to_csv``/``json.dump``/``torch.save`` in lab5_1 and lab5_2 returns
nothing), nothing was reproducible (zero ``seed_everything`` calls), and the
"results" were validation numbers on 455 samples where five of ten classes had
support <= 9. This script is the graded path: it writes a results.json record with
the split hash, the seed, the config, and the baselines the model must beat.

Run it directly for a quick look, or via slurm/train_jureca.sbatch for real runs.

    python scripts/train_cnn.py --arm sampler --seed 0 --epochs 5 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eo_course import baselines as bl  # noqa: E402
from eo_course import (  # noqa: E402
    gates,  # noqa: E402
    metrics,  # noqa: E402
    paths,
    radiometry,
    results,
    splits,
)
from eo_course import labels as lab_mod  # noqa: E402
from eo_course import patches as pat  # noqa: E402

#: Ablation arms. Each differs from ``none`` in exactly one thing, so a measured
#: difference is attributable to that thing. Names must match
#: slurm/submit_sweep.sbatch's GRID.
ARMS = {
    "none": {"imbalance": None, "note": "plain cross-entropy, no rebalancing"},
    "weights_sqrt": {"imbalance": "sqrt_inverse_freq", "note": "class weights ~ 1/sqrt(n)"},
    "weights_inv": {"imbalance": "inverse_freq", "note": "class weights ~ 1/n"},
    "sampler": {"imbalance": "weighted_sampler", "note": "WeightedRandomSampler, ~1/n"},
    "focal": {"imbalance": "focal", "note": "focal loss, gamma=2"},
}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", default="none", choices=sorted(ARMS))
    p.add_argument("--run-id", default=None)
    p.add_argument("--lab", default="lab5")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--accelerator", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--test-scenes", default=None, help="comma-separated scene ids held out for test")
    p.add_argument("--block", type=int, default=10, help="spatial block edge, in patches")
    p.add_argument("--augment", action="store_true", help="random flips on train only")
    p.add_argument("--limit-epochs", type=int, default=None, help="smoke test: run N epochs only")
    p.add_argument("--dry-run", action="store_true", help="build data + baselines + gates, skip training")
    return p


def load_and_split(args):
    """Load scenes, split by scene, fit normalisation on TRAIN ONLY."""
    ps = pat.load_all_scenes(paths.training_data_dir())
    print(f"[data] loaded {len(ps)} patches from {len(np.unique(ps.scene))} scenes")
    print(ps.summary())

    # Duplicate detector: lab5_2's glob("*_data.npz") matched per-scene files AND a
    # combined file, so every patch was loaded twice and duplicated into train+test.
    gates.gate_no_duplicate_patches(ps.patches)

    test_scenes = args.test_scenes
    if test_scenes:
        test_scenes = [s.strip() for s in test_scenes.split(",") if s.strip()]
    else:
        # Deterministic default: hold out the alphabetically-last scene so the whole
        # class shares one test set and no team can select on it.
        uniq = sorted(map(str, np.unique(ps.scene)))
        test_scenes = uniq[-1:]
        print(f"[split] no --test-scenes given; holding out {test_scenes}")

    manifest = splits.group_block_split(
        ps.scene, ps.row, ps.col, block=args.block, val_ratio=0.15,
        test_scenes=test_scenes, seed=args.seed,
    )
    gates.gate_split_is_grouped(manifest)
    gates.gate_split_disjoint(manifest)
    paths.ensure(paths.splits_dir())
    manifest.write(paths.splits_dir() / f"split_{manifest.manifest_hash}.json")
    print(f"[split] {manifest.summary()}")

    lm = lab_mod.LabelMap.from_train(ps.labels[manifest.train_idx])
    print(f"[labels] {lm}")

    # Normalisation fitted on TRAIN ONLY. lab4_2 computed percentile bounds over the
    # whole raster, which later became train+val+test: every test patch contributed
    # to the transform applied to every training patch.
    train_raw = ps.patches[manifest.train_idx]
    norm = radiometry.Norm.fit(train_raw, mode="minmax", channel_axis=-1)
    print(f"[norm] {norm}")

    x_all = norm.apply(ps.patches)
    gates.gate_input_units(x_all, "normalised patches")

    y_all = lm.encode(ps.labels)
    return ps, manifest, lm, norm, x_all, y_all


def collect_predictions(ckpt_path, dm, split="test"):
    """Load the BEST checkpoint and return (preds, targets).

    lab5_1, lab5_2 and lab6 all passed the in-memory module to ``trainer.test()``,
    which uses final-epoch weights and never consults ``ckpt_path="best"`` -- so the
    entire checkpoint-selection machinery was unexercised.
    """
    import torch

    from eo_course.training import CorineModule

    # train_counts is excluded from hparams (numpy array + weights_only load), so it
    # is passed again here. The `w` buffer is also restored from the checkpoint, but
    # being explicit avoids relying on in-place buffer copy semantics.
    module = CorineModule.load_from_checkpoint(
        str(ckpt_path), train_counts=dm.train_counts
    )
    module.eval()
    loader = dm.test_dataloader() if split == "test" else dm.val_dataloader()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(module(x).argmax(1).cpu())
            targets.append(y.cpu())
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)

    # lightning/torch are only needed to actually train. Keeping the import inside
    # the training path means `--dry-run` verifies data, split, normalisation and
    # baselines on a bare numpy install -- which is the half of the pipeline that is
    # easiest to get wrong and cheapest to check.
    pl = None
    if not args.dry_run:
        import lightning as pl_module

        pl = pl_module
        pl.seed_everything(args.seed, workers=True)

    # `<arm>_<seed>` matches notebooks_src and slurm/submit_sweep.sbatch. The
    # run id is the key in an append-only results.json, so a script run and a
    # notebook run of the same arm+seed must collide -- if they do not, the same
    # experiment lands as two unattributable records.
    run_id = args.run_id or f"{args.arm}_{args.seed}"
    print(f"=== run {run_id} | arm={args.arm} ({ARMS[args.arm]['note']}) | seed={args.seed} ===")
    print(paths.describe())

    ps, manifest, lm, norm, x_all, y_all = load_and_split(args)
    n_classes = lm.n_classes
    in_channels = x_all.shape[1] if x_all.ndim == 4 and x_all.shape[1] <= 20 else x_all.shape[-1]
    if x_all.ndim == 4 and x_all.shape[-1] == in_channels and x_all.shape[1] != in_channels:
        x_all = np.ascontiguousarray(np.transpose(x_all, (0, 3, 1, 2)))

    test_idx = manifest.test_idx
    base = bl.run_all(
        x_train=np.transpose(ps.patches[manifest.train_idx], (0, 3, 1, 2)),
        y_train_codes=ps.labels[manifest.train_idx],
        x_test=np.transpose(ps.patches[test_idx], (0, 3, 1, 2)),
        y_test_codes=ps.labels[test_idx],
        labels=np.arange(n_classes), codes=lm.codes,
        train_scene_ids=ps.scene[manifest.train_idx], test_scene_ids=ps.scene[test_idx],
        seed=args.seed,
    )
    print("\n[baselines] scored on the same test split with the same metric code")
    for name, b in sorted(base.items(), key=lambda kv: -kv[1]["macro_f1"]):
        print(f"  {name:<20} bal_acc={b['balanced_acc']:.4f} macro_f1={b['macro_f1']:.4f} "
              f"acc={b['overall_acc']:.4f}")

    if args.dry_run:
        print("\n[dry-run] skipping training")
        return 0

    # --- training -------------------------------------------------------
    import torch

    from eo_course.training import CorineDataModule, CorineModule

    dm = CorineDataModule(
        x_all, y_all, manifest, batch_size=args.batch_size,
        imbalance=ARMS[args.arm]["imbalance"], augment_train=args.augment,
        num_workers=args.num_workers, seed=args.seed,
    )
    dm.setup("fit")
    module = CorineModule(
        n_classes=n_classes, in_channels=in_channels, lr=args.lr,
        imbalance=ARMS[args.arm]["imbalance"], train_counts=dm.train_counts,
        max_epochs=args.epochs,
    )
    n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"[model] {n_params:,} trainable parameters")

    ckpt_dir = paths.ensure(paths.run_dir(run_id) / "ckpt")[0]
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    # Checkpoint on an imbalance-aware metric; stop on validation loss.
    #
    # lab5_1 logged only val_loss/val_acc, so no checkpoint could select for
    # rare-class performance, and lab5_2's EarlyStopping used mode="min" on
    # accuracy -- i.e. it stopped at the worst epoch. Both directions are fixed
    # here, but the two callbacks deliberately monitor *different* things:
    # macro-F1 over a validation split with thin per-class support is a noisy
    # quantity, so using it to decide "training is over" halts on noise. It is
    # still the right thing to select weights with.
    best = ModelCheckpoint(
        dirpath=str(ckpt_dir), monitor="val_macro_f1", mode="max",
        save_top_k=1, save_last=True, filename="best-{epoch}-{val_macro_f1:.4f}",
    )
    stop = EarlyStopping(monitor="val_loss", mode="min", patience=8)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        limit_train_batches=args.limit_epochs or 1.0,
        accelerator=args.accelerator,
        devices=args.devices,
        deterministic="warn",
        default_root_dir=str(paths.run_dir(run_id)),
        callbacks=[best, stop],
        log_every_n_steps=10,
    )
    trainer.fit(module, datamodule=dm)

    if not best.best_model_path:
        print("ERROR: no checkpoint was written; cannot evaluate the best model", file=sys.stderr)
        return 1

    # --- evaluate on TEST, once, with the best checkpoint ---------------
    dm.setup("test")
    preds, targets = collect_predictions(best.best_model_path, dm, "test")
    test_m = metrics.evaluate(
        targets, preds, np.arange(n_classes), codes=lm.codes, names=lm.names,
        groups=ps.scene[test_idx], n_boot=1000, seed=args.seed,
    )
    print("\n" + test_m.table(min_support=gates.MIN_CLASS_SUPPORT))

    record = results.record_run(
        run_id, lab=args.lab,
        config={
            "arm": args.arm, "imbalance": ARMS[args.arm]["imbalance"], "epochs": args.epochs,
            "batch_size": args.batch_size, "lr": args.lr, "n_params": int(n_params),
            "n_classes": n_classes, "in_channels": in_channels, "augment": args.augment,
            "normalisation": norm.to_dict(), "codes": list(lm.codes), "block": args.block,
            "torch": torch.__version__, "lightning": pl.__version__,
        },
        split_manifest_hash=manifest.manifest_hash, seed=args.seed,
        test_metrics=test_m.to_dict(), baselines=base,
        test_used_for_tuning=False, n_seeds=1, notes=ARMS[args.arm]["note"],
        extra={"best_checkpoint": best.best_model_path},
    )

    gates.gate_predictions_saved(
        targets, preds, paths.run_dir(run_id) / "test_predictions.npz", manifest.manifest_hash
    )

    lines = gates.run_all_gates(test_m, base, manifest, record)
    ok = gates.print_gate_board(lines)
    results.write_csv(args.lab)
    print("\n" + results.summary_table(args.lab))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
