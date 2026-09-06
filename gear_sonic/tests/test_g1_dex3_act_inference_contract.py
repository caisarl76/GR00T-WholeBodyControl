import numpy as np
import pytest

from gear_sonic.scripts.infer_g1_dex3_act import IMAGE_KEYS, load_observation


def _write(path, state=None, images=True):
    data = {"observation.state": np.zeros(28, dtype=np.float32) if state is None else state}
    if images:
        data.update({key: np.zeros((480, 640, 3), dtype=np.uint8) for key in IMAGE_KEYS})
    np.savez(path, **data)


def test_observation_contract_accepts_four_rgb_cameras(tmp_path):
    path = tmp_path / "obs.npz"
    _write(path)
    observation = load_observation(path)
    assert observation["observation.state"].shape == (28,)
    assert all(observation[key].shape == (480, 640, 3) for key in IMAGE_KEYS)


@pytest.mark.parametrize("state", [np.zeros(27), np.full(28, np.nan)])
def test_observation_contract_rejects_bad_state(tmp_path, state):
    path = tmp_path / "obs.npz"
    _write(path, state=state)
    with pytest.raises(ValueError):
        load_observation(path)


def test_observation_contract_rejects_missing_camera(tmp_path):
    path = tmp_path / "obs.npz"
    _write(path, images=False)
    with pytest.raises(ValueError):
        load_observation(path)
