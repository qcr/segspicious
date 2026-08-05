"""Tests for Phase 1: dataset naming."""

import pytest

from segspicious.datasets import (
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


# ── Base dataset names ───────────────────────────────────────────────────


class TestSyntheticDatasetName:
    def test_default_name(self):
        ds = SyntheticDataset()
        assert ds.name == "synthetic"

    def test_custom_name(self):
        ds = SyntheticDataset(name="my_dataset")
        assert ds.name == "my_dataset"


# ── Modifier name suffixes ───────────────────────────────────────────────


class TestSubsetName:
    def test_random_subset(self):
        ds = SyntheticDataset(num_samples=200, name="base")
        sub = subset(ds, n=100, seed=42)
        assert sub.name == "base-subset[n=100,seed=42]"

    def test_explicit_indices(self):
        ds = SyntheticDataset(num_samples=20, name="base")
        sub = subset(ds, indices=[0, 3, 7])
        assert sub.name == "base-subset[indices=0+3+7]"

    def test_default_seed(self):
        ds = SyntheticDataset(num_samples=20, name="base")
        sub = subset(ds, n=5, seed=0)
        assert sub.name == "base-subset[n=5,seed=0]"


class TestConcatName:
    def test_two_datasets(self):
        ds1 = SyntheticDataset(name="cityscapes_train")
        ds2 = SyntheticDataset(name="bdd100k_train")
        cat = concat_datasets([ds1, ds2])
        assert cat.name == "cityscapes_train+bdd100k_train"

    def test_three_datasets(self):
        ds1 = SyntheticDataset(name="a")
        ds2 = SyntheticDataset(name="b")
        ds3 = SyntheticDataset(name="c")
        cat = concat_datasets([ds1, ds2, ds3])
        assert cat.name == "a+b+c"

    def test_single_dataset(self):
        ds = SyntheticDataset(name="only")
        cat = concat_datasets([ds])
        assert cat.name == "only"


class TestMarkAsOodName:
    def test_single_class(self):
        ds = SyntheticDataset(num_classes=5, name="cityscapes_train")
        result = mark_as_ood(ds, classes=["class_1"])
        assert result.name == "cityscapes_train-mark_ood[class_1]"

    def test_multiple_classes_sorted(self):
        ds = SyntheticDataset(num_classes=5, name="cityscapes_train")
        result = mark_as_ood(ds, classes=["class_3", "class_1"])
        assert result.name == "cityscapes_train-mark_ood[class_1+class_3]"

    def test_by_index(self):
        ds = SyntheticDataset(
            class_names=("bicycle", "car", "motorcycle"), name="cityscapes_train"
        )
        result = mark_as_ood(ds, classes=[2, 0])
        assert result.name == "cityscapes_train-mark_ood[bicycle+motorcycle]"


class TestHoldOutOodName:
    def test_suffix(self):
        ds = SyntheticDataset(num_classes=5, name="cityscapes_train")
        marked = mark_as_ood(ds, classes=["class_1", "class_3"])
        result = hold_out_ood(marked)
        assert result.name == "cityscapes_train-mark_ood[class_1+class_3]-hold_out_ood"


class TestHoldOutClassesName:
    def test_single_class(self):
        ds = SyntheticDataset(
            num_classes=3, num_samples=5, height=2, width=2, name="cityscapes_train"
        )
        ds._labels[:] = 0  # ensure no class_2 pixels so all samples pass
        result = hold_out_classes(ds, classes=["class_2"])
        assert result.name == "cityscapes_train-hold_out[class_2]"

    def test_multiple_classes_sorted(self):
        ds = SyntheticDataset(
            num_classes=3, num_samples=5, height=2, width=2, name="cityscapes_train"
        )
        ds._labels[:] = 0
        result = hold_out_classes(ds, classes=["class_2", "class_1"])
        assert result.name == "cityscapes_train-hold_out[class_1+class_2]"


class TestSelectClassesName:
    def test_single_class(self):
        ds = SyntheticDataset(num_classes=3, name="cityscapes_train")
        result = select_classes(ds, classes=["class_0"])
        assert result.name == "cityscapes_train-select[class_0]"

    def test_multiple_classes_sorted(self):
        ds = SyntheticDataset(
            class_names=("car", "truck", "bus"), name="cityscapes_train"
        )
        result = select_classes(ds, classes=["truck", "car"])
        assert result.name == "cityscapes_train-select[car+truck]"


class TestRemapClassesName:
    def test_simple_remap(self):
        ds = SyntheticDataset(
            class_names=("road", "sidewalk", "building"), name="cityscapes_train"
        )
        result = remap_classes(ds, mapping={"road": "ground", "sidewalk": "ground"})
        assert result.name == "cityscapes_train-remap[road=ground+sidewalk=ground]"

    def test_entries_sorted_by_key(self):
        ds = SyntheticDataset(
            class_names=("road", "sidewalk", "building"), name="cityscapes_train"
        )
        result = remap_classes(ds, mapping={"sidewalk": "ground", "road": "ground"})
        assert result.name == "cityscapes_train-remap[road=ground+sidewalk=ground]"


class TestFilterSamplesName:
    def test_suffix(self):
        ds = SyntheticDataset(name="cityscapes_train")
        result = filter_samples(
            ds,
            predicate=lambda img, lbl: True,
            label="min10pct_valid",
        )
        assert result.name == "cityscapes_train-filter[min10pct_valid]"


class TestFilterByLabelsName:
    def test_suffix(self):
        ds = SyntheticDataset(name="cityscapes_train")
        result = filter_by_labels(
            ds,
            predicate=lambda lbl: True,
            label="min10pct_valid",
        )
        assert result.name == "cityscapes_train-filter[min10pct_valid]"


# ── Chained modifiers ───────────────────────────────────────────────────


class TestChainedNames:
    def test_mark_ood_then_hold_out_ood(self):
        ds = SyntheticDataset(
            class_names=("bicycle", "car", "motorcycle"),
            name="cityscapes_train",
            num_samples=5,
            height=2,
            width=2,
        )
        ds._labels[:] = 0  # all bicycle, no OoD pixels after mark
        result = hold_out_ood(mark_as_ood(ds, classes=["bicycle", "motorcycle"]))
        assert (
            result.name
            == "cityscapes_train-mark_ood[bicycle+motorcycle]-hold_out_ood"
        )

    def test_subset_then_filter(self):
        ds = SyntheticDataset(num_samples=20, name="base")
        sub = subset(ds, n=10, seed=0)
        filtered = filter_by_labels(sub, predicate=lambda lbl: True, label="nonempty")
        assert filtered.name == "base-subset[n=10,seed=0]-filter[nonempty]"

    def test_remap_then_subset(self):
        ds = SyntheticDataset(
            class_names=("road", "sidewalk", "building"), name="cityscapes_train"
        )
        remapped = remap_classes(ds, mapping={"road": "ground", "sidewalk": "ground"})
        sub = subset(remapped, n=5, seed=1)
        assert (
            sub.name
            == "cityscapes_train-remap[road=ground+sidewalk=ground]-subset[n=5,seed=1]"
        )

    def test_concat_modified_datasets(self):
        ds1 = SyntheticDataset(name="cityscapes_train")
        ds2 = SyntheticDataset(name="bdd100k_train")
        sub1 = subset(ds1, n=5, seed=0)
        sub2 = subset(ds2, n=3, seed=0)
        cat = concat_datasets([sub1, sub2])
        assert (
            cat.name
            == "cityscapes_train-subset[n=5,seed=0]+bdd100k_train-subset[n=3,seed=0]"
        )
