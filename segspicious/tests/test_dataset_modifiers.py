"""Tests for dataset modifiers."""

import pytest
import torch

from segspicious.datasets import (
    SegmentationDataset,
    concat_datasets,
    filter_by_labels,
    filter_samples,
    hold_out_classes,
    hold_out_ood,
    mark_as_ood,
    remap_classes,
    select_classes,
    subset,
)
from segspicious.tests.helpers import SyntheticDataset

# ── Subset ────────────────────────────────────────────────────────────────


class TestSubset:
    def test_explicit_indices(self):
        ds = SyntheticDataset(num_samples=10, seed=0)
        sub = subset(ds, indices=[0, 3, 7])
        assert len(sub) == 3
        assert sub[0][0].equal(ds[0][0])
        assert sub[1][0].equal(ds[3][0])
        assert sub[2][0].equal(ds[7][0])

    def test_random_subsample(self):
        ds = SyntheticDataset(num_samples=20, seed=0)
        sub = subset(ds, n=5, seed=42)
        assert len(sub) == 5

    def test_random_subsample_reproducible(self):
        ds = SyntheticDataset(num_samples=20, seed=0)
        sub1 = subset(ds, n=5, seed=42)
        sub2 = subset(ds, n=5, seed=42)
        for i in range(5):
            assert sub1[i][0].equal(sub2[i][0])

    def test_metadata_propagated(self):
        ds = SyntheticDataset(num_classes=7)
        sub = subset(ds, n=3, seed=0)
        assert sub.num_classes == ds.num_classes
        assert sub.class_names == ds.class_names
        assert sub.ignore_index == ds.ignore_index

    def test_is_segmentation_dataset(self):
        sub = subset(SyntheticDataset(), n=3, seed=0)
        assert isinstance(sub, SegmentationDataset)

    def test_error_both_indices_and_n(self):
        with pytest.raises(ValueError, match="either"):
            subset(SyntheticDataset(), indices=[0, 1], n=2)

    def test_error_neither_indices_nor_n(self):
        with pytest.raises(ValueError, match="either"):
            subset(SyntheticDataset())

    def test_error_n_exceeds_length(self):
        with pytest.raises(ValueError, match="exceeds"):
            subset(SyntheticDataset(num_samples=5), n=10)


# ── ConcatDatasets ───────────────────────────────────────────────────────


class TestConcatDatasets:
    def test_concatenation(self):
        ds1 = SyntheticDataset(num_samples=5, seed=0)
        ds2 = SyntheticDataset(num_samples=3, seed=1)
        cat = concat_datasets([ds1, ds2])
        assert len(cat) == 8
        assert cat[0][0].equal(ds1[0][0])
        assert cat[5][0].equal(ds2[0][0])

    def test_metadata_propagated(self):
        ds1 = SyntheticDataset(num_classes=4, seed=0)
        ds2 = SyntheticDataset(num_classes=4, seed=1)
        cat = concat_datasets([ds1, ds2])
        assert cat.num_classes == 4
        assert cat.class_names == ds1.class_names
        assert cat.ignore_index == ds1.ignore_index

    def test_is_segmentation_dataset(self):
        cat = concat_datasets([SyntheticDataset()])
        assert isinstance(cat, SegmentationDataset)

    def test_error_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            concat_datasets([])

    def test_error_mismatched_num_classes(self):
        with pytest.raises(ValueError, match="num_classes"):
            concat_datasets(
                [
                    SyntheticDataset(num_classes=3),
                    SyntheticDataset(num_classes=5),
                ]
            )

    def test_error_mismatched_class_names(self):
        with pytest.raises(ValueError, match="class_names"):
            concat_datasets(
                [
                    SyntheticDataset(num_classes=3),
                    SyntheticDataset(num_classes=3, class_names=("a", "b", "c")),
                ]
            )

    def test_error_mismatched_ood_class_names(self):
        """Datasets with same ID names but different OoD names should fail."""
        ds1 = SyntheticDataset(class_names=("a", "b", "c"))
        ds2 = mark_as_ood(
            SyntheticDataset(class_names=("a", "b", "c", "ood_x")),
            classes=["ood_x"],
        )
        with pytest.raises(ValueError, match="class_names"):
            concat_datasets([ds1, ds2])


# ── get_labels / get_classes_present ─────────────────────────────────────


class TestGetLabels:
    def test_base_dataset(self):
        ds = SyntheticDataset(num_samples=5, seed=0)
        for i in range(len(ds)):
            _, expected = ds[i]
            assert ds.get_labels(i).equal(expected)

    def test_through_subset(self):
        ds = SyntheticDataset(num_samples=10, seed=0)
        sub = subset(ds, indices=[2, 5, 8])
        for i in range(len(sub)):
            _, expected = sub[i]
            assert sub.get_labels(i).equal(expected)

    def test_through_remapped(self):
        ds = SyntheticDataset(num_classes=5, num_samples=5, seed=0)
        remapped = mark_as_ood(ds, classes=["class_3"])
        for i in range(len(remapped)):
            _, expected = remapped[i]
            assert remapped.get_labels(i).equal(expected)

    def test_through_concat(self):
        ds1 = SyntheticDataset(num_samples=3, seed=0)
        ds2 = SyntheticDataset(num_samples=4, seed=1)
        cat = concat_datasets([ds1, ds2])
        for i in range(len(cat)):
            _, expected = cat[i]
            assert cat.get_labels(i).equal(expected)


class TestGetClassesPresent:
    def test_base_dataset(self):
        ds = SyntheticDataset(num_classes=3, num_samples=5, height=2, width=2)
        ds._labels[0] = 0  # only class_0
        assert ds.get_classes_present(0) == frozenset({0})

    def test_multiple_classes(self):
        ds = SyntheticDataset(num_classes=3, num_samples=5, height=2, width=2)
        ds._labels[0] = torch.tensor([[0, 1], [2, 0]])
        assert ds.get_classes_present(0) == frozenset({0, 1, 2})

    def test_through_remapped(self):
        ds = SyntheticDataset(num_classes=4, num_samples=3, height=2, width=2)
        ds._labels[0] = torch.tensor([[0, 1], [2, 3]])
        # mark class_1 as OoD: kept 0→0, 2→1, 3→2; OoD 1→3
        remapped = mark_as_ood(ds, classes=["class_1"])
        classes = remapped.get_classes_present(0)
        assert classes == frozenset({0, 1, 2, 3})

    def test_through_subset(self):
        ds = SyntheticDataset(num_classes=3, num_samples=5, height=2, width=2)
        ds._labels[0] = 0
        ds._labels[2] = torch.tensor([[1, 2], [0, 1]])
        sub = subset(ds, indices=[0, 2])
        assert sub.get_classes_present(0) == frozenset({0})
        assert sub.get_classes_present(1) == frozenset({0, 1, 2})

    def test_returns_frozenset(self):
        ds = SyntheticDataset(num_classes=3, num_samples=1)
        result = ds.get_classes_present(0)
        assert isinstance(result, frozenset)


# ── filter_by_labels ─────────────────────────────────────────────────────


class TestFilterByLabels:
    def test_always_true(self):
        ds = SyntheticDataset(num_samples=5)
        filtered = filter_by_labels(ds, predicate=lambda lbl: True, label="all")
        assert len(filtered) == 5

    def test_always_false(self):
        filtered = filter_by_labels(
            SyntheticDataset(num_samples=5),
            predicate=lambda lbl: False,
            label="none",
        )
        assert len(filtered) == 0

    def test_label_predicate(self):
        ds = SyntheticDataset(num_classes=3, num_samples=5, height=2, width=2)
        ds._labels[0] = 0
        ds._labels[1] = torch.tensor([[2, 0], [1, 2]])
        ds._labels[2] = 1
        ds._labels[3] = torch.tensor([[0, 2], [2, 1]])
        ds._labels[4] = torch.tensor([[0, 1], [1, 0]])
        # Keep only samples containing class 2
        filtered = filter_by_labels(
            ds, predicate=lambda lbl: (lbl == 2).any(), label="has_class2"
        )
        assert len(filtered) == 2  # samples 1 and 3

    def test_metadata_propagated(self):
        ds = SyntheticDataset(num_classes=6)
        filtered = filter_by_labels(ds, predicate=lambda lbl: True, label="all")
        assert filtered.num_classes == ds.num_classes
        assert filtered.class_names == ds.class_names
        assert filtered.ignore_index == ds.ignore_index

    def test_returns_segmentation_dataset(self):
        filtered = filter_by_labels(SyntheticDataset(), predicate=lambda lbl: True, label="all")
        assert isinstance(filtered, SegmentationDataset)


# ── filter_samples ───────────────────────────────────────────────────────


class TestFilterSamples:
    def test_always_true(self):
        ds = SyntheticDataset(num_samples=5)
        filtered = filter_samples(ds, predicate=lambda img, lbl: True, label="all")
        assert len(filtered) == 5

    def test_always_false(self):
        filtered = filter_samples(
            SyntheticDataset(num_samples=5),
            predicate=lambda img, lbl: False,
            label="none",
        )
        assert len(filtered) == 0

    def test_partial_filter(self):
        ds = SyntheticDataset(num_samples=20, seed=0, height=32, width=32)
        filtered = filter_samples(ds, predicate=lambda img, lbl: img.mean() > 0.5, label="bright")
        assert 0 < len(filtered) < 20

    def test_metadata_propagated(self):
        ds = SyntheticDataset(num_classes=6)
        filtered = filter_samples(ds, predicate=lambda img, lbl: True, label="all")
        assert filtered.num_classes == ds.num_classes
        assert filtered.class_names == ds.class_names
        assert filtered.ignore_index == ds.ignore_index

    def test_returns_segmentation_dataset(self):
        filtered = filter_samples(SyntheticDataset(), predicate=lambda img, lbl: True, label="all")
        assert isinstance(filtered, SegmentationDataset)


# ── select_classes ───────────────────────────────────────────────────────


class TestSelectClasses:
    def _make_dataset(self):
        """Dataset with controlled labels."""
        ds = SyntheticDataset(num_classes=3, num_samples=5, height=2, width=2)
        ds._labels[0] = 0  # only class_0
        ds._labels[1] = torch.tensor([[2, 0], [1, 2]])  # has class_2
        ds._labels[2] = 1  # only class_1
        ds._labels[3] = torch.tensor([[0, 2], [2, 1]])  # has class_2
        ds._labels[4] = torch.tensor([[0, 1], [1, 0]])  # class_0 + class_1
        return ds

    def test_keeps_samples_with_class(self):
        ds = self._make_dataset()
        result = select_classes(ds, classes=["class_2"])
        assert len(result) == 2  # samples 1 and 3

    def test_multiple_classes(self):
        ds = self._make_dataset()
        result = select_classes(ds, classes=["class_0", "class_2"])
        assert len(result) == 4  # sample 2 (only class_1) excluded

    def test_no_label_remapping(self):
        ds = self._make_dataset()
        result = select_classes(ds, classes=["class_2"])
        # Labels should be unchanged from original
        _, lbl = result[0]
        assert lbl.max() <= 2  # still original class indices

    def test_metadata_unchanged(self):
        ds = self._make_dataset()
        result = select_classes(ds, classes=["class_2"])
        assert result.num_classes == ds.num_classes
        assert result.class_names == ds.class_names
        assert result.ignore_index == ds.ignore_index

    def test_by_index(self):
        ds = self._make_dataset()
        result = select_classes(ds, classes=[2])
        assert len(result) == 2

    def test_is_segmentation_dataset(self):
        result = select_classes(SyntheticDataset(), classes=["class_0"])
        assert isinstance(result, SegmentationDataset)

    def test_error_empty(self):
        with pytest.raises(ValueError, match="empty"):
            select_classes(SyntheticDataset(), classes=[])

    def test_error_invalid_name(self):
        with pytest.raises(ValueError, match="not found"):
            select_classes(SyntheticDataset(), classes=["nonexistent"])

    def test_error_invalid_index(self):
        with pytest.raises(ValueError, match="out of range"):
            select_classes(SyntheticDataset(num_classes=3), classes=[99])


# ── mark_as_ood ──────────────────────────────────────────────────────────


class TestMarkAsOod:
    def test_metadata_updated(self):
        ds = SyntheticDataset(num_classes=5)
        result = mark_as_ood(ds, classes=["class_1", "class_3"])
        assert result.num_classes == 3
        assert result.class_names == ("class_0", "class_2", "class_4")
        assert result.ood_class_names == ("class_1", "class_3")
        assert result.all_class_names == (
            "class_0",
            "class_2",
            "class_4",
            "class_1",
            "class_3",
        )
        assert result.num_ood_classes == 2
        assert result.ignore_index == 255

    def test_by_index(self):
        result = mark_as_ood(SyntheticDataset(num_classes=5), classes=[1, 3])
        assert result.num_classes == 3
        assert result.class_names == ("class_0", "class_2", "class_4")

    def test_label_remapping(self):
        ds = SyntheticDataset(
            num_classes=5, num_samples=20, height=16, width=16, seed=42
        )
        result = mark_as_ood(ds, classes=["class_1", "class_3"])

        # kept: class_0→0, class_2→1, class_4→2
        # OoD:  class_1→3, class_3→4
        expected = {0: 0, 1: 3, 2: 1, 3: 4, 4: 2}

        for i in range(len(result)):
            orig_img, orig_lbl = ds[i]
            new_img, new_lbl = result[i]
            assert new_img.equal(orig_img), "Images should be unchanged."
            for orig_c, new_c in expected.items():
                mask = orig_lbl == orig_c
                if mask.any():
                    assert (new_lbl[mask] == new_c).all()

    def test_ood_pixels_present(self):
        ds = SyntheticDataset(
            num_classes=5, num_samples=20, height=16, width=16, seed=42
        )
        result = mark_as_ood(ds, classes=["class_1", "class_3"])
        all_labels = torch.cat([result[i][1].flatten() for i in range(len(result))])
        ood_mask = all_labels >= result.num_classes
        assert ood_mask.any(), "Should have OoD pixels from marked classes."

    def test_length_unchanged(self):
        ds = SyntheticDataset(num_samples=10)
        result = mark_as_ood(ds, classes=["class_0"])
        assert len(result) == 10

    def test_ignore_index_preserved(self):
        ds = SyntheticDataset(num_classes=3, num_samples=5, height=4, width=4)
        ds._labels[0, 0, 0] = 255
        ds._labels[0, 1, 1] = 255

        result = mark_as_ood(ds, classes=["class_2"])
        _, lbl = result[0]
        assert lbl[0, 0].item() == 255
        assert lbl[1, 1].item() == 255

    def test_is_segmentation_dataset(self):
        result = mark_as_ood(SyntheticDataset(), classes=["class_0"])
        assert isinstance(result, SegmentationDataset)

    def test_composable_with_subset(self):
        ds = SyntheticDataset(num_samples=10, num_classes=5)
        marked = mark_as_ood(ds, classes=["class_3", "class_4"])
        sub = subset(marked, n=3, seed=0)
        assert isinstance(sub, SegmentationDataset)
        assert sub.num_classes == 3
        assert len(sub) == 3

    def test_error_empty(self):
        with pytest.raises(ValueError, match="empty"):
            mark_as_ood(SyntheticDataset(), classes=[])

    def test_error_invalid_name(self):
        with pytest.raises(ValueError, match="not found"):
            mark_as_ood(SyntheticDataset(), classes=["nonexistent"])

    def test_error_invalid_index(self):
        with pytest.raises(ValueError, match="out of range"):
            mark_as_ood(SyntheticDataset(num_classes=3), classes=[99])


# ── hold_out_classes ─────────────────────────────────────────────────────


class TestHoldOutClasses:
    def _make_dataset(self):
        """Dataset with controlled labels for deterministic tests."""
        ds = SyntheticDataset(num_classes=3, num_samples=5, height=2, width=2)
        ds._labels[0] = 0                                  # all class_0
        ds._labels[1] = torch.tensor([[2, 0], [1, 2]])     # has class_2
        ds._labels[2] = 1                                  # all class_1
        ds._labels[3] = torch.tensor([[0, 2], [2, 1]])     # has class_2
        ds._labels[4] = torch.tensor([[0, 1], [1, 0]])     # class_0 + class_1
        return ds

    def test_samples_removed(self):
        ds = self._make_dataset()
        result = hold_out_classes(ds, classes=["class_2"])
        assert len(result) == 3  # samples 0, 2, 4 kept

    def test_metadata_unchanged(self):
        ds = self._make_dataset()
        result = hold_out_classes(ds, classes=["class_2"])
        assert result.num_classes == ds.num_classes
        assert result.all_class_names == ds.all_class_names
        assert result.ignore_index == ds.ignore_index

    def test_no_label_remapping(self):
        ds = self._make_dataset()
        result = hold_out_classes(ds, classes=["class_2"])
        # Sample 0 was all class_0 → still all 0 (no remap)
        assert result[0][1].equal(ds[0][1])

    def test_by_index(self):
        ds = self._make_dataset()
        result = hold_out_classes(ds, classes=[2])
        assert len(result) == 3

    def test_is_segmentation_dataset(self):
        result = hold_out_classes(self._make_dataset(), classes=["class_2"])
        assert isinstance(result, SegmentationDataset)

    def test_error_empty(self):
        with pytest.raises(ValueError, match="empty"):
            hold_out_classes(SyntheticDataset(), classes=[])

    def test_error_invalid_name(self):
        with pytest.raises(ValueError, match="not found"):
            hold_out_classes(SyntheticDataset(), classes=["nonexistent"])

    def test_error_invalid_index(self):
        with pytest.raises(ValueError, match="out of range"):
            hold_out_classes(SyntheticDataset(num_classes=3), classes=[99])


# ── hold_out_ood ──────────────────────────────────────────────────────


class TestHoldOutOod:
    def _make_ood_dataset(self):
        """Dataset with OoD labels set up via mark_as_ood."""
        ds = SyntheticDataset(num_classes=5, num_samples=6, height=2, width=2)
        ds._labels[0] = 0                                  # only class_0
        ds._labels[1] = torch.tensor([[3, 0], [1, 3]])     # has class_3
        ds._labels[2] = torch.tensor([[1, 2], [0, 1]])     # class_0,1,2
        ds._labels[3] = torch.tensor([[4, 0], [1, 4]])     # has class_4
        ds._labels[4] = 2                                  # only class_2
        ds._labels[5] = torch.tensor([[0, 1], [2, 0]])     # class_0,1,2
        return mark_as_ood(ds, classes=["class_3", "class_4"])

    def test_ood_samples_removed(self):
        ds = self._make_ood_dataset()
        result = hold_out_ood(ds)
        assert len(result) == 4  # samples 1, 3 had OoD pixels

    def test_no_ood_pixels(self):
        ds = self._make_ood_dataset()
        result = hold_out_ood(ds)
        for i in range(len(result)):
            _, lbl = result[i]
            assert (lbl < result.num_classes).all() or (lbl == result.ignore_index).all()

    def test_metadata_unchanged(self):
        ds = self._make_ood_dataset()
        result = hold_out_ood(ds)
        assert result.num_classes == ds.num_classes
        assert result.all_class_names == ds.all_class_names
        assert result.ignore_index == ds.ignore_index

    def test_no_ood_is_noop(self):
        ds = SyntheticDataset(num_classes=3, num_samples=5)
        result = hold_out_ood(ds)
        assert len(result) == 5  # no OoD pixels, nothing filtered

    def test_is_segmentation_dataset(self):
        result = hold_out_ood(self._make_ood_dataset())
        assert isinstance(result, SegmentationDataset)


# ── mark_as_ood + hold_out_ood consistency ─────────────────────────────


class TestMarkAsOodAndHoldOutConsistency:
    """mark_as_ood (eval) and mark_as_ood + hold_out_ood (train) are two
    sides of the same operation. Both must produce the same metadata and
    label mapping for shared samples."""

    def _make_dataset(self):
        ds = SyntheticDataset(num_classes=5, num_samples=6, height=2, width=2)
        ds._labels[0] = 0                                  # only class_0
        ds._labels[1] = torch.tensor([[3, 0], [1, 3]])     # has class_3
        ds._labels[2] = torch.tensor([[1, 2], [0, 1]])     # class_0,1,2
        ds._labels[3] = torch.tensor([[4, 0], [1, 4]])     # has class_4
        ds._labels[4] = 2                                  # only class_2
        ds._labels[5] = torch.tensor([[0, 1], [2, 0]])     # class_0,1,2
        return ds

    def test_same_metadata(self):
        ds = self._make_dataset()
        ood_classes = ["class_3", "class_4"]
        eval_ds = mark_as_ood(ds, classes=ood_classes)
        train_ds = hold_out_ood(mark_as_ood(ds, classes=ood_classes))
        assert eval_ds.num_classes == train_ds.num_classes
        assert eval_ds.all_class_names == train_ds.all_class_names
        assert eval_ds.ignore_index == train_ds.ignore_index

    def test_same_labels_on_shared_samples(self):
        ds = self._make_dataset()
        ood_classes = ["class_3", "class_4"]
        eval_ds = mark_as_ood(ds, classes=ood_classes)
        train_ds = hold_out_ood(mark_as_ood(ds, classes=ood_classes))

        # train keeps samples 0, 2, 4, 5 (no OoD pixels)
        # eval keeps all 6 — compare on the shared samples
        train_labels = [train_ds[i][1] for i in range(len(train_ds))]
        shared_eval_indices = [0, 2, 4, 5]
        for train_i, eval_i in enumerate(shared_eval_indices):
            assert eval_ds[eval_i][1].equal(train_labels[train_i])


# ── remap_classes ────────────────────────────────────────────────────────


class TestRemapClasses:
    def test_merge_classes(self):
        ds = SyntheticDataset(num_classes=5)
        remapped = remap_classes(
            ds,
            mapping={
                "class_0": "group_a",
                "class_1": "group_a",
                "class_2": "group_b",
                "class_3": "group_b",
                "class_4": "group_c",
            },
        )
        assert remapped.num_classes == 3

    def test_label_remapping(self):
        ds = SyntheticDataset(num_classes=5, num_samples=10, height=8, width=8, seed=42)
        remapped = remap_classes(
            ds,
            mapping={
                "class_0": "group_a",
                "class_1": "group_a",
                "class_2": "group_b",
                "class_3": "group_b",
                "class_4": "group_c",
            },
        )
        # class_0, class_1 → 0;  class_2, class_3 → 1;  class_4 → 2
        expected = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}
        for i in range(len(remapped)):
            orig_img, orig_lbl = ds[i]
            rem_img, rem_lbl = remapped[i]
            assert rem_img.equal(orig_img)
            for old_c, new_c in expected.items():
                mask = orig_lbl == old_c
                if mask.any():
                    assert (rem_lbl[mask] == new_c).all()

    def test_unmapped_stay_id(self):
        ds = SyntheticDataset(
            num_classes=5, num_samples=20, height=16, width=16, seed=42
        )
        remapped = remap_classes(
            ds,
            mapping={
                "class_0": "kept_0",
                "class_1": "kept_1",
            },
        )
        # 2 mapped + 3 unmapped = 5 ID classes
        assert remapped.num_classes == 5
        assert remapped.class_names == (
            "kept_0", "kept_1", "class_2", "class_3", "class_4",
        )
        assert remapped.ood_class_names == ()
        for i in range(len(remapped)):
            _, lbl = remapped[i]
            # All labels should be valid ID classes
            assert (lbl < 5).all()

    def test_class_names_from_mapping(self):
        ds = SyntheticDataset(num_classes=5)
        remapped = remap_classes(
            ds,
            mapping={
                "class_0": "group_a",
                "class_1": "group_a",
                "class_2": "group_b",
                "class_3": "group_b",
                "class_4": "group_c",
            },
        )
        assert remapped.class_names == ("group_a", "group_b", "group_c")

    def test_class_name_order_is_first_appearance(self):
        ds = SyntheticDataset(num_classes=4)
        remapped = remap_classes(
            ds,
            mapping={
                "class_0": "second",
                "class_1": "first",
                "class_2": "first",
                "class_3": "second",
            },
        )
        assert remapped.class_names == ("second", "first")

    def test_identity_remap(self):
        ds = SyntheticDataset(num_classes=3)
        remapped = remap_classes(
            ds,
            mapping={
                "class_0": "class_0",
                "class_1": "class_1",
                "class_2": "class_2",
            },
        )
        assert isinstance(remapped, SegmentationDataset)
        assert remapped.num_classes == 3
        assert remapped.class_names == ds.class_names

    def test_existing_ood_preserved(self):
        """Existing OoD classes survive remapping."""
        base = SyntheticDataset(
            num_samples=5,
            height=2,
            width=2,
            class_names=("road", "car", "building", "motorcycle"),
        )
        # Ensure sample 0 has motorcycle pixels so mark_as_ood has
        # something to remap.
        base._labels[0] = torch.tensor([[3, 0], [1, 3]])
        ds = mark_as_ood(base, classes=["motorcycle"])

        remapped = remap_classes(
            ds,
            mapping={
                "road": "ground",
                "car": "ground",
                "building": "structure",
            },
        )
        assert remapped.num_classes == 2
        assert remapped.class_names == ("ground", "structure")
        assert remapped.ood_class_names == ("motorcycle",)

        _, lbl = remapped[0]
        # motorcycle(3) → 2 (shifted OoD), road(0) → 0, car(1) → 0
        expected = torch.tensor([[2, 0], [0, 2]])
        assert lbl.equal(expected)

    def test_error_empty_mapping(self):
        with pytest.raises(ValueError, match="empty"):
            remap_classes(SyntheticDataset(), mapping={})

    def test_error_invalid_class_name(self):
        with pytest.raises(ValueError, match="not found"):
            remap_classes(
                SyntheticDataset(num_classes=3),
                mapping={"nonexistent": "a"},
            )
