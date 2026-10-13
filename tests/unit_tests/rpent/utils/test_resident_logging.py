from rpent.utils.logging import get_logger, init_output_dir, reset_output_logging


def test_sequential_runs_do_not_write_to_the_previous_log(tmp_path):
    reset_output_logging()
    try:
        first = init_output_dir(tmp_path / "first")
        get_logger("resident-test").info("first-only-marker")
        reset_output_logging()
        second = init_output_dir(tmp_path / "second")
        get_logger("resident-test").info("second-only-marker")
        reset_output_logging()
        assert "first-only-marker" in (first / "run.log").read_text()
        assert "second-only-marker" not in (first / "run.log").read_text()
        assert "second-only-marker" in (second / "run.log").read_text()
        assert "first-only-marker" not in (second / "run.log").read_text()
    finally:
        reset_output_logging()
