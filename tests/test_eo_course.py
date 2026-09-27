"""Tests for the eo_course toolkit.

These run anywhere (no cluster, no data download) and are the same tests students
must pass in Lab 2 before they are allowed to train anything. They also encode the
bugs from the 2025/26 edition as executable regressions: if someone reintroduces
``np.unique``-derived label sets or a positional npz read, a test fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from eo_course import baselines as bl
from eo_course import gates, labels, metrics, radiometry, results, splits


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def synthetic():
    """A small imbalanced 4-class problem with 3 scenes and a spatial grid."""
    rng = np.random.default_rng(0)
    n_per_scene = 300
    scenes, rows, cols, y = [], [], [], []
    codes = [1, 12, 23, 41]
    prior = [0.5, 0.3, 0.18, 0.02]
    for s in range(3):
        for i in range(n_per_scene):
            scenes.append(f"S2A_20180{s + 4}15")
            rows.append(i // 20)
            cols.append(i % 20)
            y.append(rng.choice(codes, p=prior))
    y = np.asarray(y, dtype=np.int64)
    return {
        "labels": y,
        "scene": np.asarray(scenes, dtype=object),
        "row": np.asarray(rows, dtype=np.int32),
        "col": np.asarray(cols, dtype=np.int32),
        "codes": codes,
    }


@pytest.fixture
def preds(synthetic):
    """A 'model' that is better than chance but imperfect, plus a constant one."""
    lm = labels.LabelMap(tuple(synthetic["codes"]))
    y_true = lm.encode(synthetic["labels"])
    rng = np.random.default_rng(7)
    y_pred = y_true.copy()
    flip = rng.random(y_true.size) < 0.15
    y_pred[flip] = rng.integers(0, lm.n_classes, size=int(flip.sum()))
    const = np.zeros_like(y_true)
    return {"labels": np.arange(lm.n_classes), "y_true": y_true, "y_good": y_pred, "y_const": const}


# --------------------------------------------------------------------------
# radiometry
# --------------------------------------------------------------------------
class TestRadiometry:
    def test_dn_roundtrip(self):
        dn = np.array([0, 1000, 4000, 10000], dtype=np.uint16)
        r = radiometry.dn_to_reflectance(dn)
        assert r.dtype == np.float32
        assert np.allclose(r, [0.0, 0.1, 0.4, 1.0])
        assert np.array_equal(radiometry.reflectance_to_dn(r), dn)

    def test_prithvi_stats_are_the_verified_ones(self):
        # Guard against a future edit inventing numbers. Verified from
        # terratorch/models/backbones/prithvi_vit.py
        assert radiometry.PRITHVI_V2_MEAN == (1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0)
        assert radiometry.PRITHVI_V2_STD == (2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0)
        assert radiometry.PRITHVI_V2_BANDS == ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2")

    def test_prithvi_norm_4band_is_prefix_of_6band(self):
        m6, s6 = radiometry.prithvi_norm(6)
        m4, s4 = radiometry.prithvi_norm(4)
        assert m4 == m6[:4] and s4 == s6[:4]
        with pytest.raises(ValueError):
            radiometry.prithvi_norm(8)

    def test_per_band_percentiles_are_per_band(self):
        # Blue dim, NIR bright: a pooled percentile would crush Blue toward a constant.
        data = np.stack([np.full((20, 20), 100.0), np.full((20, 20), 9000.0)]).astype(np.float32)
        low, high = radiometry.per_band_percentiles(data)
        assert low.shape == (2,)
        assert low[0] < 200 and low[1] > 8000, "percentiles must be computed per band"

    def test_norm_must_be_fitted(self):
        n = radiometry.Norm(mode="percentile")
        assert not n.fitted
        with pytest.raises(RuntimeError, match="has not been fitted"):
            n.apply(np.random.rand(4, 3, 3, 2))

    def test_norm_percentile_is_per_band_and_bounded(self):
        rng = np.random.default_rng(1)
        train = np.concatenate([rng.normal(200, 20, (50, 8, 8, 2)), rng.normal(8000, 100, (50, 8, 8, 2))])
        n = radiometry.Norm.fit(train, mode="percentile", channel_axis=-1)
        out = n.apply(train)
        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0
        # Each band must independently span roughly [0,1]; a pooled stretch would not.
        for c in range(2):
            assert out[..., c].std() > 0.1

    def test_norm_zscore_zero_std_band_is_safe(self):
        train = np.zeros((10, 4, 4, 2), dtype=np.float32)
        train[..., 1] = 5.0
        n = radiometry.Norm.fit(train, mode="zscore", channel_axis=-1)
        out = n.apply(train)
        assert np.all(np.isfinite(out))

    def test_norm_zscore_is_not_clipped_to_positive(self):
        """A z-score is signed; clipping it to (0, 1) deletes half the data.

        Regression test: ``clip`` used to default to ``(0.0, 1.0)`` for every
        mode, so ``Norm(mode="zscore")`` mapped every below-mean pixel to
        exactly 0.0 -- 50% of a symmetric distribution, silently.
        """
        rng = np.random.default_rng(7)
        train = rng.normal(1000.0, 300.0, size=(40, 6, 6, 3)).astype(np.float32)
        n = radiometry.Norm.fit(train, mode="zscore", channel_axis=-1)
        out = n.apply(train)
        assert n.clip == (-3.0, 3.0)
        assert out.min() < -1.0, "negative z-scores were clipped away"
        assert (out == 0.0).mean() < 0.05, "too many values collapsed to exactly zero"
        # clip=None must still mean "do not clip at all".
        raw = radiometry.Norm.fit(train, mode="zscore", channel_axis=-1, clip=None)
        assert raw.clip is None
        assert raw.apply(train).max() > 3.0

    def test_norm_clip_defaults_are_mode_specific(self):
        train = np.random.rand(20, 5, 5, 2).astype(np.float32) * 4000
        assert radiometry.Norm.fit(train, mode="minmax").clip == (0.0, 1.0)
        assert radiometry.Norm.fit(train, mode="percentile").clip == (0.0, 1.0)
        assert radiometry.Norm.fit(train, mode="none").clip is None

    def test_norm_dict_roundtrip(self):
        train = np.random.rand(20, 6, 6, 3).astype(np.float32)
        n = radiometry.Norm.fit(train, mode="percentile", channel_axis=-1)
        m = radiometry.Norm.from_dict(n.to_dict())
        assert np.allclose(m.apply(train), n.apply(train))

    def test_assert_reflectance_range_catches_both_failure_modes(self):
        # DN masquerading as reflectance (lab4_1 wrote raw DN through).
        with pytest.raises(AssertionError, match="not reflectance"):
            radiometry.assert_reflectance_range(np.random.rand(5, 5) * 9000)
        # Double-normalised data (lab5_1's `* 0.0001`).
        with pytest.raises(AssertionError, match="implausibly small"):
            radiometry.assert_reflectance_range(np.random.rand(5, 5) * 1e-4)
        radiometry.assert_reflectance_range(np.random.rand(5, 5))  # ok


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------
class TestLabels:
    def test_taxonomy_has_44_classes_and_no_phantom_48(self):
        # The 2025/26 notebooks declared 48: "No data" and printed "45 classes".
        assert len(labels.CORINE_CLASSES) == 44
        assert 48 not in labels.CORINE_CLASSES
        assert labels.CORINE_NODATA == -128

    def test_water_excluded_by_default(self):
        default = labels.valid_class_codes()
        assert 44 not in default and 43 not in default and 30 not in default
        assert 44 in labels.valid_class_codes(exclude=())
        with pytest.raises(ValueError):
            labels.valid_class_codes(exclude=(999,))

    def test_clean_labels_rejects_nodata_and_water(self):
        arr = np.array([[-128, 0, 1, 12, 44, 45]])
        mask = labels.clean_labels(arr)
        assert mask.tolist() == [[False, False, True, True, False, False]]

    def test_clean_labels_rejects_float(self):
        with pytest.raises(TypeError):
            labels.clean_labels(np.array([1.0, 2.0]))

    def test_labelmap_encode_decode_roundtrip(self):
        lm = labels.LabelMap((1, 12, 23, 41))
        assert lm.n_classes == 4
        codes = np.array([1, 12, 23, 41, 1, 41])
        enc = lm.encode(codes)
        assert enc.dtype == np.int64
        assert enc.tolist() == [0, 1, 2, 3, 0, 3]
        assert lm.decode(enc).tolist() == codes.tolist()

    def test_labelmap_strict_on_unknown_code(self):
        lm = labels.LabelMap((1, 12, 23))
        with pytest.raises(ValueError, match="not in the LabelMap"):
            lm.encode(np.array([1, 999]))

    def test_labelmap_rejects_duplicates_and_empty(self):
        with pytest.raises(ValueError):
            labels.LabelMap((1, 1, 2))
        with pytest.raises(ValueError):
            labels.LabelMap(())

    def test_from_train_ignores_test_only_classes(self):
        lm = labels.LabelMap.from_train(np.array([1, 1, 12, 12, -128]))
        assert lm.codes == (1, 12)
        with pytest.raises(ValueError, match="not in the LabelMap"):
            lm.encode(np.array([23]))

    def test_class_counts_keeps_absent_classes_as_zero_rows(self):
        c = labels.class_counts(np.array([1, 1, 12]), [1, 12, 23, 41])
        assert c.tolist() == [2, 1, 0, 0]

    def test_imbalance_ratio_is_max_over_min_not_positional(self):
        # lab4_2 used counts[0]/counts[-1], which is only max/min if sorted.
        assert labels.imbalance_ratio(np.array([1, 100, 50])) == 100.0
        with pytest.raises(ValueError):
            labels.imbalance_ratio(np.array([0, 0]))

    def test_level1_grouping(self):
        assert labels.level1_of(1) == "Artificial surfaces"
        assert labels.level1_of(12) == "Agricultural areas"
        assert labels.level1_of(23) == "Forest and semi-natural areas"
        assert labels.level1_of(36) == "Wetlands"
        assert labels.level1_of(44) == "Water bodies"
        with pytest.raises(ValueError):
            labels.level1_of(48)


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
class TestSplits:
    def test_group_block_split_holds_out_whole_scenes(self, synthetic):
        m = splits.group_block_split(
            synthetic["scene"], synthetic["row"], synthetic["col"],
            block=4, val_ratio=0.2, test_scenes=["S2A_20180415"], seed=0,
        )
        assert m.group_leakage() == {"train&val": set(), "train&test": set(), "val&test": set()}
        assert m.overlaps() == {"train&val": set(), "train&test": set(), "val&test": set()}
        test_scenes = set(synthetic["scene"][m.test_idx].tolist())
        assert test_scenes == {"S2A_20180415"}

    def test_split_is_deterministic_in_seed(self, synthetic):
        a = splits.group_block_split(synthetic["scene"], block=4, seed=3)
        b = splits.group_block_split(synthetic["scene"], block=4, seed=3)
        c = splits.group_block_split(synthetic["scene"], block=4, seed=4)
        assert a.manifest_hash == b.manifest_hash
        assert a.manifest_hash != c.manifest_hash

    def test_too_few_scenes_is_an_error_not_a_fallback(self):
        with pytest.raises(RuntimeError, match="at least 3 distinct scenes"):
            splits.group_block_split(np.array(["A", "A", "B"] * 20))

    def test_manifest_roundtrip(self, synthetic, tmp_path):
        m = splits.group_block_split(synthetic["scene"], block=4, seed=0)
        p = m.write(tmp_path / "split.json")
        m2 = splits.SplitManifest.read(p)
        assert m2.manifest_hash == m.manifest_hash
        m.verify_against(m2)
        with pytest.raises(RuntimeError, match="mismatch"):
            m.verify_against(splits.group_block_split(synthetic["scene"], block=4, seed=9))

    def test_stratified_random_is_available_but_flagged(self, synthetic):
        m = splits.stratified_random_split(synthetic["labels"], seed=0)
        assert "leak" in m.extra["warning"]
        # GateFailure subclasses AssertionError, not RuntimeError.
        with pytest.raises(gates.GateFailure, match="negative control"):
            gates.gate_split_is_grouped(m)

    def test_get_split_negative_train_bug_is_gone(self):
        # 2025/26: a class with 1 sample gave num_train = -1 and the assert passed
        # because -1 + 1 + 1 == 1, putting the same sample in val and test.
        y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
        with pytest.raises(RuntimeError, match="cannot form a split"):
            splits.stratified_random_split(y, min_per_class=2)

    def test_support_check_rejects_thin_test_split(self, synthetic):
        with pytest.raises(RuntimeError, match="under-supported"):
            splits.group_block_split(
                synthetic["scene"], block=4, seed=0, labels=synthetic["labels"], min_per_class=25,
            )

    def test_block_ids_shape_and_range(self):
        rows = np.repeat(np.arange(20), 20)
        cols = np.tile(np.arange(20), 20)
        b = splits.block_ids(rows, cols, 20, 20, block=5)
        assert b.shape == (400,)
        assert b.min() == 0 and b.max() == 15  # 4x4 super-cells

    def test_block_larger_than_grid_is_rejected(self):
        with pytest.raises(ValueError, match="smaller than block"):
            splits.block_ids(np.arange(9), np.arange(9), 3, 3, block=10)

    def test_per_split_class_counts(self, synthetic):
        m = splits.group_block_split(synthetic["scene"], block=4, seed=0)
        counts = splits.per_split_class_counts(synthetic["labels"], m, synthetic["codes"])
        assert sum(counts["train"]) + sum(counts["val"]) + sum(counts["test"]) == 900
        assert counts["codes"] == synthetic["codes"]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
class TestMetrics:
    def test_balanced_accuracy_floor_is_one_over_k(self, preds):
        m = metrics.evaluate(preds["y_true"], preds["y_const"], preds["labels"], n_boot=0)
        assert m.balanced_acc == pytest.approx(0.25)
        assert m.floor_balanced_acc() == pytest.approx(0.25)

    def test_absent_class_still_occupies_a_row(self):
        # The 2025/26 bug: labels = np.unique(concat(true,pred)) dropped absent
        # classes from the average, inflating balanced accuracy.
        y_true = np.array([0, 0, 0, 1, 1, 2])
        y_pred = np.array([0, 0, 0, 1, 1, 1])  # class 3 never appears anywhere
        m = metrics.evaluate(y_true, y_pred, np.arange(4), n_boot=0)
        assert m.support[3] == 0
        assert m.confusion.shape == (4, 4)
        assert m.balanced_acc == pytest.approx((1.0 + 1.0 + 0.0) / 3)  # over classes with support

    def test_matches_sklearn(self):
        from sklearn.metrics import balanced_accuracy_score, f1_score

        rng = np.random.default_rng(3)
        for _ in range(20):
            yt = rng.integers(0, 5, 200)
            yp = rng.integers(0, 5, 200)
            m = metrics.evaluate(yt, yp, np.arange(5), n_boot=0)
            assert m.balanced_acc == pytest.approx(balanced_accuracy_score(yt, yp))
            assert m.macro_f1 == pytest.approx(f1_score(yt, yp, average="macro", zero_division=0))
            assert m.weighted_f1 == pytest.approx(f1_score(yt, yp, average="weighted", zero_division=0))

    def test_rejects_class_outside_fixed_label_set(self):
        with pytest.raises(ValueError, match="absent from the fixed"):
            metrics.evaluate(np.array([0, 1, 5]), np.array([0, 1, 1]), np.arange(3), n_boot=0)

    def test_shape_mismatch_and_empty(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            metrics.evaluate(np.array([0, 1]), np.array([0]), np.arange(2), n_boot=0)
        with pytest.raises(ValueError, match="empty"):
            metrics.evaluate(np.array([]), np.array([]), np.arange(2), n_boot=0)

    def test_bootstrap_ci_is_clustered_when_groups_given(self, preds):
        groups = np.repeat(np.arange(10), preds["y_true"].size // 10)
        m = metrics.evaluate(preds["y_true"], preds["y_good"], preds["labels"], groups=groups, n_boot=100)
        assert m.ci["clustered"] is True
        lo, hi = m.ci["balanced_acc"]
        assert lo <= m.balanced_acc <= hi

    def test_clustered_ci_is_wider_than_iid(self, preds):
        # Correlated patches: an i.i.d. bootstrap is falsely reassuring.
        y = preds["y_true"]
        yp = preds["y_good"]
        # Build strongly scene-dependent errors so clustering matters.
        scene = np.repeat(np.arange(20), y.size // 20)
        iid = metrics.bootstrap_ci(y, yp, preds["labels"], groups=None, n_boot=300)
        clu = metrics.bootstrap_ci(y, yp, preds["labels"], groups=scene, n_boot=300)
        w_iid = iid["balanced_acc"][1] - iid["balanced_acc"][0]
        w_clu = clu["balanced_acc"][1] - clu["balanced_acc"][0]
        assert iid["clustered"] is False and clu["clustered"] is True
        assert w_clu >= w_iid * 0.5  # clustering must not narrow the interval

    def test_paired_bootstrap_detects_a_real_gap(self, preds):
        np.random.default_rng(11)
        better = preds["y_good"]
        worse = preds["y_const"]
        out = metrics.paired_bootstrap_delta(preds["y_true"], better, worse, preds["labels"], n_boot=300)
        assert out["significant_balanced_acc"] is True
        assert out["mean_delta_balanced_acc"] > 0

    def test_paired_bootstrap_rejects_noise(self, preds):
        rng = np.random.default_rng(12)
        a = preds["y_good"].copy()
        b = a.copy()
        flip = rng.random(a.size) < 0.005
        b[flip] = rng.integers(0, 4, int(flip.sum()))
        out = metrics.paired_bootstrap_delta(preds["y_true"], a, b, preds["labels"], n_boot=300)
        assert out["significant_balanced_acc"] is False

    def test_merge_rare_classes(self):
        y = np.array([1, 1, 1, 1, 12, 12, 12, 23])
        y2, codes, mapping = metrics.merge_rare_classes(y, [1, 12, 23], min_support=3)
        assert 23 not in codes and mapping["merged"] == [23]
        assert (y2 == mapping["rare_other_code"]).sum() == 1

    def test_top_confusions_names_the_absorbing_class(self, preds):
        m = metrics.evaluate(preds["y_true"], preds["y_const"], preds["labels"],
                             codes=[1, 12, 23, 41], n_boot=0)
        top = metrics.top_confusions(m, k=3)
        assert top and top[0][2] > 0
        assert top[0][1] == m.class_label(0)

    def test_table_flags_thin_support(self, preds):
        m = metrics.evaluate(preds["y_true"], preds["y_good"], preds["labels"], n_boot=0)
        txt = m.table(min_support=200)
        assert "n<200" in txt or "NO SUPPORT" in txt


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
class TestBaselines:
    def test_majority_balanced_accuracy_is_one_over_k(self, synthetic):
        # A constant predictor scores 1/K balanced accuracy ONLY when every class
        # is present in the truth set. Evaluate against real labels, not a
        # single-class array, or you get 1.0 and learn nothing.
        pred = bl.majority_class(synthetic["labels"], synthetic["codes"], len(synthetic["labels"]))
        lm = labels.LabelMap(tuple(synthetic["codes"]))
        y_true = lm.encode(synthetic["labels"])
        assert len(np.unique(y_true)) == 4, "fixture must contain all classes"
        m = metrics.evaluate(y_true, pred, np.arange(4), n_boot=0)
        assert m.balanced_acc == pytest.approx(0.25)
        assert m.floor_balanced_acc() == pytest.approx(0.25)
        # ...while its overall accuracy equals the majority class rate, which is
        # the comparison people actually get wrong.
        assert m.overall_acc == pytest.approx(m.majority_rate())

    def test_uniform_random_is_at_chance(self, synthetic):
        pred = bl.uniform_random(4000, 4, seed=0)
        rng = np.random.default_rng(5)
        truth = rng.integers(0, 4, 4000)
        m = metrics.evaluate(truth, pred, np.arange(4), n_boot=0)
        assert m.balanced_acc == pytest.approx(0.25, abs=0.02)

    def test_prior_random_beats_uniform_on_accuracy_not_balance(self, synthetic):
        truth = synthetic["labels"][:500]
        lm = labels.LabelMap(tuple(synthetic["codes"]))
        t = lm.encode(truth)
        p_prior = bl.prior_matched_random(synthetic["labels"], synthetic["codes"], 500, seed=0)
        p_unif = bl.uniform_random(500, 4, seed=0)
        assert metrics.evaluate(t, p_prior, np.arange(4), n_boot=0).overall_acc > \
               metrics.evaluate(t, p_unif, np.arange(4), n_boot=0).overall_acc

    def test_per_scene_majority_uses_scene_histogram(self):
        tr_scene = np.array(["A"] * 10 + ["B"] * 10)
        tr_lab = np.array([1] * 8 + [12] * 2 + [23] * 9 + [1])
        te_scene = np.array(["A", "B", "Z"])
        pred = bl.per_scene_majority(te_scene, tr_scene, tr_lab, [1, 12, 23])
        assert pred.tolist() == [0, 2, 0]  # A->1, B->23, unseen Z -> global majority 1

    def test_ndvi_rule_separates_vegetation_from_bare(self):
        # (N, C, H, W) with B04 (red) at index 2 and B08 (nir) at index 3.
        x = np.zeros((2, 4, 8, 8), dtype=np.float32)
        x[0, 2] = 0.1
        x[0, 3] = 0.5     # high NDVI -> vegetation
        x[1, 2] = 0.3
        x[1, 3] = 0.31    # ~zero NDVI -> bare/urban
        codes = [1, 12, 23]
        pred = bl.ndvi_rule(x, codes)
        names = labels.LabelMap(tuple(codes)).names
        assert names[pred[0]] == "Broad-leaved forest"
        assert names[pred[1]] == "Continuous urban fabric"

    def test_linear_probe_learns_a_separable_problem(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, (120, 3, 8, 8)).astype(np.float32)
        y = np.tile([0, 1], 60)                      # interleaved, so both classes
        x[:, 0, 4, 4] += np.where(y == 1, 3.0, 0.0)  # appear in train AND test
        perm = rng.permutation(120)
        x, y = x[perm], y[perm]
        predict, report = bl.linear_probe(x[:90], y[:90], np.arange(2))
        m = metrics.evaluate(y[90:], predict(x[90:]), np.arange(2), n_boot=0)
        assert min(m.support) > 0, "test split must contain both classes"
        assert m.balanced_acc > 0.8
        assert report["type"] == "logistic_regression"

    def test_run_all_returns_scored_baselines(self, synthetic):
        lm = labels.LabelMap(tuple(synthetic["codes"]))
        out = bl.run_all(
            x_train=None, y_train_codes=synthetic["labels"][:600],
            x_test=None, y_test_codes=synthetic["labels"][600:],
            labels=np.arange(lm.n_classes), codes=synthetic["codes"],
            train_scene_ids=synthetic["scene"][:600], test_scene_ids=synthetic["scene"][600:],
            which=("majority", "uniform_random", "prior_random", "per_scene_majority"),
        )
        assert set(out) == {"majority", "uniform_random", "prior_random", "per_scene_majority"}
        assert all("macro_f1" in v and "balanced_acc" in v for v in out.values())


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------
class TestGates:
    def test_collapsed_model_is_caught(self, preds):
        m = metrics.evaluate(preds["y_true"], preds["y_const"], preds["labels"], n_boot=0)
        with pytest.raises(gates.GateFailure, match="collapsed"):
            gates.gate_not_collapsed(m)

    def test_chance_model_is_caught(self, preds):
        m = metrics.evaluate(preds["y_true"], preds["y_const"], preds["labels"], n_boot=0)
        with pytest.raises(gates.GateFailure, match="1/K"):
            gates.gate_above_chance(m)

    def test_beats_baselines_requires_margin(self, preds, synthetic):
        m = metrics.evaluate(preds["y_true"], preds["y_const"], preds["labels"], n_boot=0)
        base = {"majority": {"macro_f1": 0.1, "balanced_acc": 0.25}}
        with pytest.raises(gates.GateFailure, match="does not beat"):
            gates.gate_beats_baselines(m, base)
        with pytest.raises(gates.GateFailure, match="no baselines"):
            gates.gate_beats_baselines(m, {})

    def test_good_model_passes_performance_gates(self, preds):
        m = metrics.evaluate(preds["y_true"], preds["y_good"], preds["labels"], n_boot=0)
        base = {"majority": {"macro_f1": 0.05, "balanced_acc": 0.25}}
        gates.gate_above_chance(m)
        gates.gate_not_collapsed(m)
        gates.gate_beats_baselines(m, base)

    def test_input_units_gate(self):
        rng = np.random.default_rng(0)
        # Default is the strict course convention: reflectance in [0, 1].
        with pytest.raises(gates.GateFailure):
            gates.gate_input_units(rng.random((4, 4, 4, 2)) * 8000)
        with pytest.raises(gates.GateFailure):
            gates.gate_input_units(rng.random((4, 4, 4, 2)) * 1e-4)
        gates.gate_input_units(rng.random((4, 4, 4, 2)))

    def test_input_units_gate_accepts_standardized(self):
        """Prithvi wants z-scored input; the gate must not reject it as 'wrong units'."""
        rng = np.random.default_rng(1)
        z = rng.normal(0.0, 1.0, size=(64, 8, 8, 4)).astype(np.float32)
        gates.gate_input_units(z, "z", expect="standardized")
        # ...but the strict default still rejects it, so a notebook cannot pass
        # the reflectance gate by accident while feeding standardized data.
        with pytest.raises(gates.GateFailure):
            gates.gate_input_units(z, "z")

    def test_input_units_gate_rejects_pathological_in_every_mode(self):
        rng = np.random.default_rng(2)
        tiny = rng.random((8, 4, 4, 2)) * 1e-6
        flat = np.full((8, 4, 4, 2), 0.5)
        for expect in ("auto", "standardized", "dn"):
            with pytest.raises(gates.GateFailure):
                gates.gate_input_units(tiny, "tiny", expect=expect)
            with pytest.raises(gates.GateFailure):
                gates.gate_input_units(flat, "flat", expect=expect)

    def test_input_units_gate_dn_mode(self):
        rng = np.random.default_rng(3)
        gates.gate_input_units(rng.random((8, 4, 4, 2)) * 8000, "dn", expect="dn")
        with pytest.raises(gates.GateFailure):
            gates.gate_input_units(rng.random((8, 4, 4, 2)), "already scaled", expect="dn")

    def test_input_units_gate_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="unknown expect"):
            gates.gate_input_units(np.random.default_rng(0).random((4, 4)), expect="furlongs")

    def test_duplicate_patch_gate(self):
        uniq = np.random.default_rng(0).normal(size=(50, 3, 3, 4)).astype(np.float32)
        gates.gate_no_duplicate_patches(uniq)
        dup = np.concatenate([uniq, uniq])
        with pytest.raises(gates.GateFailure, match="duplicate patches"):
            gates.gate_no_duplicate_patches(dup)

    def test_npz_key_gate(self):
        class Fake:
            files = ["arr_0", "arr_1"]
        with pytest.raises(gates.GateFailure, match="missing required keys"):
            gates.gate_npz_loaded_by_key(Fake())

    def test_test_set_selection_gate(self):
        with pytest.raises(gates.GateFailure, match="test_used_for_tuning"):
            gates.gate_no_test_set_selection({"run_id": "r", "test_used_for_tuning": True})
        gates.gate_no_test_set_selection({"run_id": "r", "test_used_for_tuning": False})

    def test_split_hash_gate(self, synthetic):
        m = splits.group_block_split(synthetic["scene"], block=4, seed=0)
        other = splits.group_block_split(synthetic["scene"], block=4, seed=1)
        gates.gate_split_hash_matches({"run_id": "r", "split_manifest_hash": m.manifest_hash}, m)
        with pytest.raises(gates.GateFailure, match="recorded split_manifest_hash"):
            gates.gate_split_hash_matches({"run_id": "r", "split_manifest_hash": other.manifest_hash}, m)

    @staticmethod
    def _rec(v):
        """Minimal results record carrying one balanced accuracy."""
        return {"test": {"balanced_acc": v}}

    def test_seed_gate_demands_three(self, preds):
        rec = self._rec
        with pytest.raises(gates.GateFailure, match="need >= 3 seeds"):
            gates.gate_seeds_and_variance([rec(0.5), rec(0.52)])
        st = gates.gate_seeds_and_variance([rec(0.5), rec(0.52), rec(0.51)])
        assert st["n"] == 3 and st["threshold"] > 0

    def test_claim_must_exceed_seed_noise(self, preds):
        rec = self._rec
        runs = [rec(0.50), rec(0.56), rec(0.53)]
        with pytest.raises(gates.GateFailure, match=r"below 2\.0 x pooled std"):
            gates.gate_claim_supported(0.01, runs, label="focal loss")
        gates.gate_claim_supported(0.5, runs, label="real effect")

    def test_gate_board_reports_all_lines(self, synthetic, preds):
        m = splits.group_block_split(synthetic["scene"], block=4, seed=0)
        met = metrics.evaluate(preds["y_true"][: len(m.test_idx)] if False else
                               labels.LabelMap(tuple(synthetic["codes"])).encode(synthetic["labels"][m.test_idx]),
                               np.zeros(len(m.test_idx), dtype=np.int64),
                               np.arange(4), n_boot=0)
        rec = {"run_id": "r", "split_manifest_hash": m.manifest_hash, "test_used_for_tuning": False}
        lines = gates.run_all_gates(met, {"majority": {"macro_f1": 0.1, "balanced_acc": 0.25}}, m, rec)
        assert len(lines) == 8
        assert any("[FAIL]" in ln for ln in lines)


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------
class TestResults:
    def test_record_and_read(self, tmp_path, preds):
        m = metrics.evaluate(preds["y_true"], preds["y_good"], preds["labels"], n_boot=0)
        p = tmp_path / "results.json"
        rec = results.record_run(
            "run_a", lab="lab5", config={"lr": 1e-3, "n_params": 1234},
            split_manifest_hash="abc123", seed=0, test_metrics=m.to_dict(),
            baselines={"majority": {"macro_f1": 0.05, "balanced_acc": 0.25}}, path=p,
        )
        assert rec["test"]["macro_f1"] == pytest.approx(m.macro_f1)
        loaded = results.load_results(p)["runs"]
        assert len(loaded) == 1
        assert loaded[0]["config"]["n_params"] == 1234

    def test_run_id_is_append_only(self, tmp_path, preds):
        m = metrics.evaluate(preds["y_true"], preds["y_good"], preds["labels"], n_boot=0)
        p = tmp_path / "results.json"
        kw = {"lab": "lab5", "config": {}, "split_manifest_hash": "a", "seed": 0, "test_metrics": m.to_dict(), "path": p}
        results.record_run("dup", **kw)
        with pytest.raises(results.ResultsError, match="append-only"):
            results.record_run("dup", **kw)

    def test_empty_test_metrics_is_rejected(self, tmp_path):
        with pytest.raises(results.ResultsError, match="test_metrics is empty"):
            results.record_run("r", lab="l", config={}, split_manifest_hash=None, seed=0,
                               test_metrics={}, path=tmp_path / "r.json")

    def test_missing_metric_key_is_rejected(self, tmp_path):
        with pytest.raises(results.ResultsError, match="missing required key"):
            results.record_run("r", lab="l", config={}, split_manifest_hash=None, seed=0,
                               test_metrics={"n": 10}, path=tmp_path / "r.json")

    def test_numpy_types_are_serialisable(self, tmp_path, preds):
        m = metrics.evaluate(preds["y_true"], preds["y_good"], preds["labels"], n_boot=50)
        p = tmp_path / "r.json"
        results.record_run("np", lab="l", config={"np_int": np.int64(3), "np_arr": np.arange(3)},
                           split_manifest_hash="x", seed=np.int64(0), test_metrics=m.to_dict(), path=p)
        assert results.get_run("np", p)["config"]["np_int"] == 3

    def test_summary_table_and_csv(self, tmp_path, preds):
        m = metrics.evaluate(preds["y_true"], preds["y_good"], preds["labels"], n_boot=0)
        p = tmp_path / "r.json"
        for s in (0, 1, 2):
            results.record_run(f"r{s}", lab="lab5", config={"n_params": 100},
                               split_manifest_hash="h", seed=s, test_metrics=m.to_dict(),
                               baselines={"majority": {"macro_f1": 0.05, "balanced_acc": 0.25}}, path=p)
        txt = results.summary_table("lab5", p)
        assert txt.count("lab5") == 0 and "r0" in txt and "r2" in txt
        csv = results.write_csv("lab5", p, tmp_path / "r.csv")
        assert len(csv.read_text(encoding="utf-8").strip().splitlines()) == 4

    def test_gate_results_file_requires_baselines(self, tmp_path, preds):
        m = metrics.evaluate(preds["y_true"], preds["y_good"], preds["labels"], n_boot=0)
        p = tmp_path / "r.json"
        results.record_run("nb", lab="lab9", config={}, split_manifest_hash="h", seed=0,
                           test_metrics=m.to_dict(), path=p)
        with pytest.raises(gates.GateFailure, match="no baselines"):
            gates.gate_results_file(p, lab="lab9")
        with pytest.raises(gates.GateFailure, match="no results recorded"):
            gates.gate_results_file(p, lab="lab_missing")
