from pathlib import Path
import time

from learning import LEARNING_BASE_DIR, PRIORITIES_DIR, RECORDS_DIR
from learning.utils import change_dir_mod


class Config:
    """Configuration for a run."""

    def __init__(
        self,
        website: str,
        scenario: str,
        user_id,
        group_id,
        run_time="",
        run_name="run",
        repeat=1,
        priomode="chromeext",
        run_dir="",
        bwdown=10,
        rtt=100,
        bdp=1,
        loss=0.0,
    ):
        self.user_id = user_id
        self.group_id = group_id
        self.website = website
        self.website_resources = RECORDS_DIR / self.website
        self.website_priorities = PRIORITIES_DIR / (self.website + ".csv")
        self.run_data_dir = LEARNING_BASE_DIR / "run_data"
        self.data_dir = LEARNING_BASE_DIR / "run_data" / scenario / self.website
        self.data_dir.mkdir(parents=True, exist_ok=True)
        change_dir_mod(LEARNING_BASE_DIR / "run_data", self.user_id, self.group_id)
        if not run_time:
            run_time = time.strftime("%Y%m%d-%Hh%Mm%Ss")
        self.run_time = run_time
        self.run_name = run_name + "_" + run_time
        self.namespace = run_name
        if not run_dir:
            self.run_dir = self.data_dir / self.run_name
        else:
            self.run_dir = Path(run_dir)

        self.evalconfig = {
            "repeat": repeat,
            "website": self.website,
            "resources": str(self.website_resources),
            "priorities": str(self.data_dir / self.run_name / "priorities.csv"),
            "cc": "cubic",
            "bwdown": bwdown,
            "rtt": rtt,
            "bdp": bdp,
            "loss": loss,
            "priomode": priomode,
        }
