from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOSTS = "7.150.12.45,7.150.14.170"


def read(relative_path):
    return (ROOT / relative_path).read_text()


def test_training_launchers_default_to_two_nodes():
    expected = {
        "scripts/launch_multinode_torchrun.sh": (
            'NNODES="${NNODES:-2}"',
            f'HOSTS_CSV="${{HOSTS:-{DEFAULT_HOSTS}}}"',
            'MASTER_ADDR="${MASTER_ADDR:-7.150.12.45}"',
        ),
        "scripts/pretrain_pipeline.sh": (
            'NNODES="${NNODES:-2}"',
            f'HOSTS="${{HOSTS:-{DEFAULT_HOSTS}}}"',
            'MASTER_ADDR="${MASTER_ADDR:-${HOSTS%%,*}}"',
        ),
        "scripts/pretrain_3node.sh": (
            "set_default NNODES 2",
            f'set_default HOSTS "{DEFAULT_HOSTS}"',
            "set_default MASTER_ADDR 7.150.12.45",
        ),
    }

    for relative_path, snippets in expected.items():
        source = read(relative_path)
        for snippet in snippets:
            assert snippet in source, f"{relative_path} is missing {snippet!r}"


def test_formal_and_smoke_training_wrappers_use_two_node_topology():
    for scale in ("100M", "500M", "1B"):
        formal_path = f"work_record/step3_model_{scale}.sh"
        formal_source = read(formal_path)
        assert f"HOSTS=${{HOSTS:-{DEFAULT_HOSTS}}}" in formal_source
        assert "MASTER_ADDR=${MASTER_ADDR:-${HOSTS%%,*}}" in formal_source
        assert "NNODES=${NNODES:-2}" in formal_source

        smoke_path = f"work_record/step2_mode_test_{scale}.sh"
        smoke_source = read(smoke_path)
        assert f"HOSTS={DEFAULT_HOSTS}" in smoke_source
        assert "MASTER_ADDR=7.150.12.45" in smoke_source
        assert "NNODES=2" in smoke_source


def test_example_environment_uses_two_node_topology():
    source = read(".env.example")
    assert "NNODES=2" in source
    assert "MASTER_ADDR=7.150.12.45" in source
    assert f"HOSTS={DEFAULT_HOSTS}" in source
