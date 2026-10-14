from mi_bici import config


def test_paths_are_under_project_root():
    assert config.PROJ_ROOT.joinpath("pyproject.toml").is_file()
    for path in (config.DATA_DIR, config.MODELS_DIR, config.REPORTS_DIR):
        assert path.parent == config.PROJ_ROOT
