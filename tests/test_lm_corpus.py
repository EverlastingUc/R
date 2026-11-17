import json

import pytest

from entropy_lm.corpus import prepare_corpus


def _write_corpus(root, documents):
    root.mkdir(parents=True, exist_ok=True)
    for name, content in documents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def test_prepare_corpus_is_deterministic_and_splits_documents(tmp_path):
    source = tmp_path / "source"
    _write_corpus(
        source,
        {
            "a.txt": "alpha training document",
            "nested/b.txt": "beta validation document",
            "c.txt": "gamma held out document",
            "d.txt": "delta fourth document",
            "e.txt": "epsilon fifth document",
        },
    )
    first = prepare_corpus(source, tmp_path / "first", seed=17, vocab_size=270)
    second = prepare_corpus(source, tmp_path / "second", seed=17, vocab_size=270)
    assert first["documents"] == second["documents"]
    assert first["splits"] == second["splits"]
    assert first["tokenizer_fingerprint"] == second["tokenizer_fingerprint"]
    assert (tmp_path / "first" / "train.jsonl").read_text() == (
        tmp_path / "second" / "train.jsonl"
    ).read_text()


def test_exact_duplicates_are_removed_before_splitting(tmp_path):
    source = tmp_path / "source"
    _write_corpus(
        source,
        {
            "one.txt": "same exact document",
            "two.txt": "same exact document",
            "three.txt": "another document",
            "four.txt": "third document",
        },
    )
    manifest = prepare_corpus(source, tmp_path / "out", tokenizer_type="byte")
    assert manifest["document_count"] == 3
    assert manifest["duplicate_document_count"] == 1
    duplicate_record = next(
        item for item in manifest["documents"] if item["duplicate_sources"]
    )
    assert duplicate_record["duplicate_sources"] == ["two.txt"]


def test_bpe_tokenizer_is_fit_on_training_documents_only(tmp_path):
    source = tmp_path / "source"
    _write_corpus(
        source,
        {
            "a.txt": "TRAINING_ONLY_WORD " * 5,
            "b.txt": "validation phrase",
            "c.txt": "test phrase",
            "d.txt": "fourth phrase",
            "e.txt": "fifth phrase",
        },
    )
    manifest = prepare_corpus(source, tmp_path / "out", seed=0, vocab_size=280)
    training_sources = {
        item["source"] for item in manifest["documents"] if item["split"] == "train"
    }
    training_text = "".join(
        (source / name).read_text(encoding="utf-8") for name in training_sources
    )
    assert "TRAINING_ONLY_WORD" in training_text
    assert manifest["tokenizer"]["type"] == "byte-bpe"
    assert len(manifest["tokenizer"]["merges"]) <= 280 - 258


def test_manifest_records_hashes_and_tokenizer_fingerprint(tmp_path):
    source = tmp_path / "docs"
    _write_corpus(source, {"doc.txt": "hello corpus"})
    manifest = prepare_corpus(source, tmp_path / "prepared", tokenizer_type="byte")
    saved = json.loads((tmp_path / "prepared" / "manifest.json").read_text())
    assert saved == manifest
    assert len(manifest["documents"][0]["sha256"]) == 64
    assert len(manifest["tokenizer_fingerprint"]) == 64
    record = json.loads((tmp_path / "prepared" / "train.jsonl").read_text())
    assert record["tokenizer_fingerprint"] == manifest["tokenizer_fingerprint"]
    assert record["token_ids"]


def test_empty_files_are_reported_and_not_used(tmp_path):
    source = tmp_path / "docs"
    _write_corpus(source, {"empty.txt": "  \n", "real.txt": "usable"})
    manifest = prepare_corpus(source, tmp_path / "prepared", tokenizer_type="byte")
    assert manifest["document_count"] == 1
    assert manifest["skipped_files"] == [
        {"path": "empty.txt", "reason": "empty_or_whitespace"}
    ]


def test_invalid_utf8_is_rejected(tmp_path):
    source = tmp_path / "docs"
    source.mkdir()
    (source / "broken.txt").write_bytes(b"hello\xff")
    with pytest.raises(ValueError, match="valid UTF-8"):
        prepare_corpus(source, tmp_path / "prepared")


def test_invalid_split_fractions_and_tokenizer_are_rejected(tmp_path):
    source = tmp_path / "docs"
    _write_corpus(source, {"doc.txt": "a document"})
    with pytest.raises(ValueError, match="sum to less"):
        prepare_corpus(
            source,
            tmp_path / "a",
            validation_fraction=0.6,
            test_fraction=0.4,
        )
    with pytest.raises(ValueError, match="tokenizer_type"):
        prepare_corpus(source, tmp_path / "b", tokenizer_type="unknown")


def test_single_file_input_and_output_inside_source_are_safe(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text("single source document", encoding="utf-8")
    manifest = prepare_corpus(source, tmp_path / "output", tokenizer_type="byte")
    assert manifest["document_count"] == 1

    folder = tmp_path / "folder"
    _write_corpus(folder, {"a.txt": "first", "b.txt": "second"})
    prepare_corpus(folder, folder / "prepared", tokenizer_type="byte")
    assert not any(
        item["source"].startswith("prepared/") for item in json.loads(
            (folder / "prepared" / "manifest.json").read_text()
        )["documents"]
    )


def test_split_has_no_document_overlap(tmp_path):
    source = tmp_path / "docs"
    _write_corpus(source, {f"{index}.txt": f"document {index}" for index in range(12)})
    manifest = prepare_corpus(source, tmp_path / "prepared", seed=91)
    assignments = [item["sha256"] for item in manifest["documents"]]
    assert len(assignments) == len(set(assignments))
    assert sum(manifest["splits"].values()) == manifest["document_count"]
    assert manifest["splits"]["train"] > 0
