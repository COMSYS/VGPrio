from pathlib import Path

import pandas as pd

from testbed import BASE_DIR as TESTBED_BASE_DIR
from testbed.setup.process import run


def read_priofile(file_path):
    """reads a prio file and returns a dataframe with the columns: src, type, priority, ffclass, ffweight"""
    df = pd.read_csv(file_path, delimiter="#", header=None)
    return df


def write_priofile(file_path, df: pd.DataFrame):
    """writes a dataframe to a prio file with the columns: src, type, priority, ffclass, ffweight"""
    df_copy = df.copy()
    df_copy["src"] = "GET:" + df_copy["src"].astype(str)
    df_copy.to_csv(
        file_path,
        sep="#",
        index=False,
        header=False,
        lineterminator="",
        columns=[
            "src",
            "type",
            "priority",
            "incremental",
            "ffclass",
            "ffweight",
        ],
    )


def prio_switch(prio: int) -> str:
    """converts the int parameter to the actual EPS priority class"""
    return {
        1: "highest",
        2: "high",
        3: "normal",
        4: "low",
        5: "lowest",
        6: "idle",
        7: "throttled",
    }.get(prio, "Unknown")


def change_dir_mod(dir: Path, user_id: int, group_id: int):
    """gives the user and group full access to the directory and its contents"""
    run(["chmod", "777", "-R", f"{dir}"])
    run(["chown", f"{user_id}:{group_id}", "-R", f"{dir}"])


def get_all_websites() -> list[str]:
    """returns a list of all websites in the testbed"""
    sites = []
    path = Path(f"{TESTBED_BASE_DIR}/mm/record")
    for subdir in path.iterdir():
        sites.append(subdir.name)
    sites.sort()
    return sites
