"""Check that inference chunks never carry an old prompt across a boundary."""

from gear_sonic.scripts.evaluate_gr00t_subtasks_open_loop import query_intervals


def test_query_intervals_covers_every_frame_and_exact_off_grid_boundaries():
    prompts = ["approach"] * 23 + ["grasp"] * 3 + ["pick"] * 22
    chunks = list(query_intervals(prompts, 20))
    assert chunks == [(0, 20), (20, 23), (23, 26), (26, 46), (46, 48)]
    assert [frame for start, end in chunks for frame in range(start, end)] == list(range(len(prompts)))
    assert all(len(set(prompts[start:end])) == 1 for start, end in chunks)
