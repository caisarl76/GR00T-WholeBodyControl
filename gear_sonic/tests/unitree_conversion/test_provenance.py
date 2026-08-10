import hashlib
from pathlib import Path

import pytest

from gear_sonic.data.unitree_conversion import provenance as provenance_module
from gear_sonic.data.unitree_conversion.contracts import ArtifactSpec, SourceSpec
from gear_sonic.data.unitree_conversion.provenance import (
    discover_collection_lock,
    load_source_lock,
    materialize_artifact,
    source_lock_sha256,
    stratified_episode_ids,
    verify_file,
)
from gear_sonic.scripts.lock_unitree_unifolm_sources import write_discovered_lock

LOCK = Path("gear_sonic/data/unitree_conversion/manifests/smoke_sources.yaml")
DEX3_COLLECTION = "unitreerobotics/unifolm-g1-dex3-dataset"
WBT_COLLECTION = "unitreerobotics/unifolm-wbt-dataset"


class FakeHfApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.collections = {
            DEX3_COLLECTION: self._collection(
                self._item("dataset", "unitreerobotics/dex3-first"),
                self._item("model", "unitreerobotics/not-a-dataset"),
                self._item("dataset", "unitreerobotics/dex3-second"),
            ),
            WBT_COLLECTION: self._collection(self._item("dataset", "unitreerobotics/wbt-first")),
        }
        self.revisions = {
            "unitreerobotics/dex3-first": "a" * 40,
            "unitreerobotics/dex3-second": "B" * 40,
            "unitreerobotics/wbt-first": "c" * 40,
        }

    @staticmethod
    def _item(item_type: str, item_id: str):
        return type(
            "CollectionItem",
            (),
            {"item_type": item_type, "item_id": item_id},
        )()

    @staticmethod
    def _collection(*items):
        return type("Collection", (), {"items": list(items)})()

    def get_collection(self, collection_slug: str):
        self.calls.append(("get_collection", collection_slug))
        return self.collections[collection_slug]

    def dataset_info(self, repo_id: str):
        self.calls.append(("dataset_info", repo_id))
        return type(
            "DatasetInfo",
            (),
            {"id": repo_id, "sha": self.revisions[repo_id]},
        )()


def test_smoke_lock_is_fully_immutable() -> None:
    lock = load_source_lock(LOCK)

    assert lock.encoder.revision == "9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2"
    assert lock.encoder.sha256 == "60be43157f57d812f38bdbb740a5de5d5d070e8840d9edc16f02a91a6d06255b"
    assert lock.dex3.episodes == (0, 78, 155, 233, 310)
    assert lock.inspire.episodes == (0, 152, 304, 456, 608)
    for source in lock.sources:
        assert len(source.revision) == 40
        assert int(source.revision, 16) >= 0
    with pytest.raises(TypeError):
        lock.dex3.camera_map["new.camera"] = "new.target"


def test_stratified_selection_matches_pinned_counts() -> None:
    assert stratified_episode_ids(311) == (0, 78, 155, 233, 310)
    assert stratified_episode_ids(609) == (0, 152, 304, 456, 608)


def test_stratified_selection_uses_half_up_rounding() -> None:
    assert stratified_episode_ids(3) == (0, 1, 1, 2, 2)


@pytest.mark.parametrize("episode_count", [0, -1, True, 1.5])
def test_stratified_selection_rejects_invalid_counts(episode_count) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        stratified_episode_ids(episode_count)


def test_load_source_lock_rejects_floating_revision(tmp_path: Path) -> None:
    lock_text = LOCK.read_text().replace("c9552eb3b1cb610cd6227555e9e98b1bde826a77", "main")
    candidate = tmp_path / "floating.yaml"
    candidate.write_text(lock_text)

    with pytest.raises(ValueError, match="40-character immutable revision"):
        load_source_lock(candidate)


def test_load_source_lock_rejects_unknown_fields(tmp_path: Path) -> None:
    candidate = tmp_path / "unknown.yaml"
    candidate.write_text(f"{LOCK.read_text()}unexpected: true\n")

    with pytest.raises(ValueError, match="unknown fields.*unexpected"):
        load_source_lock(candidate)


def test_verify_file_checks_size_before_hashing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"wrong")

    def fail_if_hashed():
        raise AssertionError("hashing must not start after a size mismatch")

    monkeypatch.setattr(provenance_module.hashlib, "sha256", fail_if_hashed)
    with pytest.raises(ValueError, match="size mismatch"):
        verify_file(artifact, expected_size=6, expected_sha256="0" * 64)


def test_verify_file_hashes_in_one_mib_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"x" * (1024 * 1024 + 7))
    chunks: list[int] = []

    class RecordingHash:
        def update(self, data: bytes) -> None:
            chunks.append(len(data))

        @staticmethod
        def hexdigest() -> str:
            return "a" * 64

    monkeypatch.setattr(provenance_module.hashlib, "sha256", RecordingHash)

    resolved = verify_file(
        artifact,
        expected_size=1024 * 1024 + 7,
        expected_sha256="a" * 64,
    )

    assert chunks == [1024 * 1024, 7]
    assert resolved == artifact.resolve()


def test_verify_file_rejects_sha256_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"wrong")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_file(artifact, expected_size=5, expected_sha256="0" * 64)


def test_materialize_artifact_pins_revision_and_verifies_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"model")
    calls: list[dict[str, str]] = []
    spec = ArtifactSpec(
        repo_id="nvidia/GEAR-SONIC",
        revision="1" * 40,
        filename="low_latency/model_encoder.onnx",
        size=5,
        sha256=hashlib.sha256(b"model").hexdigest(),
    )

    def downloader(**kwargs: str) -> str:
        calls.append(kwargs)
        return str(artifact)

    resolved = materialize_artifact(spec, downloader=downloader)

    assert calls == [
        {
            "repo_id": "nvidia/GEAR-SONIC",
            "filename": "low_latency/model_encoder.onnx",
            "revision": "1" * 40,
        }
    ]
    assert resolved == artifact.resolve()


def test_collection_discovery_resolves_every_dataset_to_sha_in_order() -> None:
    api = FakeHfApi()

    proposed = discover_collection_lock(
        api=api,
        collection_slugs=(DEX3_COLLECTION, WBT_COLLECTION),
    )
    smoke_lock = load_source_lock(LOCK)

    assert proposed.scope == "full"
    assert proposed.encoder == smoke_lock.encoder
    assert proposed.observation_config == smoke_lock.observation_config
    assert [source.repo_id for source in proposed.sources] == [
        "unitreerobotics/dex3-first",
        "unitreerobotics/dex3-second",
        "unitreerobotics/wbt-first",
    ]
    assert [source.revision for source in proposed.sources] == [
        "a" * 40,
        "B" * 40,
        "c" * 40,
    ]
    assert all(source.approved is False for source in proposed.sources)
    assert [membership.slug for membership in proposed.collections] == [
        DEX3_COLLECTION,
        WBT_COLLECTION,
    ]
    assert proposed.collections[0].repo_ids == (
        "unitreerobotics/dex3-first",
        "unitreerobotics/dex3-second",
    )
    assert api.calls == [
        ("get_collection", DEX3_COLLECTION),
        ("dataset_info", "unitreerobotics/dex3-first"),
        ("dataset_info", "unitreerobotics/dex3-second"),
        ("get_collection", WBT_COLLECTION),
        ("dataset_info", "unitreerobotics/wbt-first"),
    ]


def test_collection_discovery_rejects_duplicate_repositories() -> None:
    api = FakeHfApi()
    api.collections[WBT_COLLECTION] = api._collection(api._item("dataset", "unitreerobotics/dex3-first"))

    with pytest.raises(ValueError, match="duplicate dataset repository"):
        discover_collection_lock(
            api=api,
            collection_slugs=(DEX3_COLLECTION, WBT_COLLECTION),
        )


def test_discovery_draft_round_trips_and_overwrite_is_fail_closed(
    tmp_path: Path,
) -> None:
    lock = discover_collection_lock(
        api=FakeHfApi(),
        collection_slugs=(DEX3_COLLECTION, WBT_COLLECTION),
    )
    output_lock = tmp_path / "review.yaml"

    assert write_discovered_lock(lock, output_lock) == output_lock.resolve()
    assert load_source_lock(output_lock) == lock

    output_lock.write_text("reviewed lock\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_discovered_lock(lock, output_lock)
    assert output_lock.read_text() == "reviewed lock\n"

    assert write_discovered_lock(lock, output_lock, force=True) == output_lock.resolve()
    assert load_source_lock(output_lock) == lock
    assert list(tmp_path.glob(f".{output_lock.name}.*.tmp")) == []


@pytest.mark.parametrize(
    ("revision", "sha256", "message"),
    [
        ("g" * 40, "0" * 64, "40-character immutable revision"),
        ("0" * 39, "0" * 64, "40-character immutable revision"),
        ("0" * 40, "z" * 64, "64-character SHA-256"),
    ],
)
def test_artifact_contract_rejects_malformed_hashes(revision: str, sha256: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ArtifactSpec(
            repo_id="owner/repository",
            revision=revision,
            filename="artifact.bin",
            size=1,
            sha256=sha256,
        )


def test_source_contract_allows_unapproved_discovery_draft() -> None:
    source = SourceSpec(
        approved=False,
        repo_id="owner/repository",
        revision="a" * 40,
    )

    assert source.episode_count is None
    assert source.episodes == ()
    assert source.primary_camera is None
    assert dict(source.camera_map) == {}


@pytest.mark.parametrize(
    ("episode_count", "episodes", "primary_camera", "camera_map", "message"),
    [
        (0, (), "observation.images.cam", {"observation.images.cam": "ego"}, "episode_count"),
        (3, (0, 3), "observation.images.cam", {"observation.images.cam": "ego"}, "episode ID"),
        (3, (0, 2), "", {"observation.images.cam": "ego"}, "primary_camera"),
        (3, (0, 2), "missing", {"observation.images.cam": "ego"}, "camera_map"),
    ],
)
def test_approved_source_contract_rejects_invalid_episode_or_camera_data(
    episode_count,
    episodes,
    primary_camera,
    camera_map,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SourceSpec(
            approved=True,
            repo_id="owner/repository",
            revision="a" * 40,
            episode_count=episode_count,
            episodes=episodes,
            primary_camera=primary_camera,
            camera_map=camera_map,
        )


def test_source_lock_sha256_hashes_exact_manifest_bytes() -> None:
    assert source_lock_sha256(LOCK) == hashlib.sha256(LOCK.read_bytes()).hexdigest()
