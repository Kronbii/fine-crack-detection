from fine_tracing import config


def test_default_sort_method():
    assert config.sort_method in {"classic", "MST", "greedy"}
