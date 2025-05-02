from pathlib import Path

from testbed import BASE_DIR as TESTBED_BASE_DIR

# learning paths
LEARNING_BASE_DIR = Path(__file__).resolve().parent

# testbed paths
RECORDS_DIR = TESTBED_BASE_DIR / "mm" / "record"
PRIORITIES_DIR = TESTBED_BASE_DIR / "mm" / "priorities"
