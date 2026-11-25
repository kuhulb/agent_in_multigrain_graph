def pytest_configure(config):
    if config.option.basetemp is None:
        cache_dir = config.rootpath / ".cache"
        cache_dir.mkdir(exist_ok=True)
        config.option.basetemp = str(cache_dir / "pytest_tmp")
