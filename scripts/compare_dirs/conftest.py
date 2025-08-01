from pathlib import Path

import pytest


def pytest_generate_tests(metafunc):
    """Dynamically generate parametrized tests based on files in directories."""
    # Check if the test function uses the required fixtures
    if "ref_file" in metafunc.fixturenames and "curr_file" in metafunc.fixturenames:
        # Get the directory paths from the command-line options
        ref_dir_str = metafunc.config.getoption("reference_dir")
        curr_dir_str = metafunc.config.getoption("current_dir")

        ref_dir = Path(ref_dir_str)
        curr_dir = Path(curr_dir_str)

        # Basic validation
        if not ref_dir.is_dir():
            pytest.fail(f"Reference directory not found: {ref_dir}")
        if not curr_dir.is_dir():
            pytest.fail(f"Current directory not found: {curr_dir}")

        # Find all pairs of files to compare
        file_pairs = []
        # Walk through the reference directory
        for ref_file in ref_dir.rglob("*"):
            if ref_file.is_file():
                # Find the corresponding file in the current directory
                relative_path = ref_file.relative_to(ref_dir)
                curr_file = curr_dir / relative_path

                if curr_file.exists() and curr_file.is_file():
                    file_pairs.append((ref_file, curr_file))
                else:
                    print(f"Warning: File {curr_file} not found for comparison.")

        if not file_pairs:
            pytest.skip("No comparable file pairs found.")

        # Create human-readable IDs for each test case
        ids = [str(ref.relative_to(ref_dir)) for ref, curr in file_pairs]

        # Parametrize the test with the found file pairs
        metafunc.parametrize("ref_file,curr_file", file_pairs, ids=ids)


@pytest.fixture
def rtol(request):
    return request.config.getoption("--rtol")


@pytest.fixture
def atol(request):
    return request.config.getoption("--atol")


def pytest_addoption(parser):
    """Add custom command-line options for directories to pytest."""
    parser.addoption(
        "--reference-dir", action="store", required=True, help="Path to the reference directory."
    )
    parser.addoption(
        "--current-dir",
        action="store",
        required=True,
        help="Path to the current directory to compare.",
    )
    parser.addoption("--rtol", type=float, help="Relative tolerance for numerical comparisons.")
    parser.addoption("--atol", type=float, help="Absolute tolerance for numerical comparisons.")
