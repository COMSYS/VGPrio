import json
import logging
from pathlib import Path
import time

if __name__ == "__main__":
    import sys

    # Add parent folder to sys.path
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from learning import LEARNING_BASE_DIR
from learning.objectives import cluster_training
from testbed import BASE_DIR as TESTBED_BASE_DIR


def train_website(params: dict):
    """
    complete training suite, running bayesian optimization to optimize the ObservedSpeedIndex of the website and evaluates
    against chrome and firefox priorization.

    Args:
        website (str): the website to be optimized
        n_iter (int): number of iterations for the bayesian optimization
        reps (int): number of repetitions for the bayesian optimization
        run_name (str): name of the run, also used for network namespaces
    """
    website = params.get("website")
    scenario = params.get("scenario")
    uid = params.get("uid")
    gid = params.get("gid")
    n_iter = params.get("n_iter")
    reps = params.get("reps")
    ch_reps = params.get("ch_reps")
    run_name = params.get("run_name")
    eval_reps = params.get("eval_reps")
    log_tb = params.get("log_tb")
    run = params.get("run")
    bwdown = params.get("bwdown")
    rtt = params.get("rtt")
    bdp = params.get("bdp")
    loss = params.get("loss")
    incremental = params.get("incremental")
    if not run:
        run = cluster_training(
            website,
            scenario,
            uid,
            gid,
            init_points=1,
            n_iter=n_iter,
            ch_reps=ch_reps,
            repeat=reps,
            run_name=run_name,
            log_tb=log_tb,
            incremental=incremental,
            bwdown=bwdown,
            rtt=rtt,
            bdp=bdp,
            loss=loss,
        )
        run.run_static_mode(mode="chrome", reps=eval_reps)
        run.run_static_mode(mode="firefox", reps=eval_reps)
        run.run_static_mode(mode="rr", reps=eval_reps)
        run.run_static_mode(mode="priofile", reps=eval_reps)

    run.plot_iterations()
    run.plot_mode_speeds()


if __name__ == "__main__":
    timestr = time.strftime("%Y%m%d-%Hh%Mm%Ss")

    logger = logging.getLogger(f"training-{timestr}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.FileHandler(LEARNING_BASE_DIR / "run_data/current_training.log")
    handler.setLevel(logging.INFO)
    with open(LEARNING_BASE_DIR / "run_data/current_training.log", "w"):
        pass  # clears log file
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s-%(name)s-%(levelname)s]: %(message)s")
    handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(handler)

    # launches the training (for shell script usage)
    with open("learning/conf.json", "r") as f:
        config = json.load(f)

    mm_path = TESTBED_BASE_DIR / "mm" / "record"

    split_list_n = lambda x,n: [x[int(i/n*len(x)):int((i+1)/n*len(x))]for i in range(n)]

    # Usage:
    subdirectories = [
        str(subdir.name) for subdir in mm_path.iterdir() if subdir.is_dir()
    ]
    subdirectories.sort()
    print(subdirectories)

    parts = split_list_n(subdirectories, config["parts"])

    with open(LEARNING_BASE_DIR / "scenarios.json", "r") as f:
        scenarios = json.load(f)

    runconfigs = []
    try:
        with open(LEARNING_BASE_DIR / "runs.jsonl", "r") as f:
            runconfigs = [json.loads(line) for line in f]
    except:
        pass

    Debug = config["debug"]
    scenario_by_name = {scenario["name"]:scenario for scenario in scenarios}

    if len(runconfigs) == 0:
        # For restarts after crash
        start_website = config["start_website"]
        start_scenario = config["start_scenario"]
        if start_website and start_scenario:
            restart = True
        else:
            restart = False

        for scenario in scenarios:
            if restart:
                if scenario["name"] != start_scenario:
                    continue
            for website in parts[config["part"] - 1]:
                if restart:
                    if website != start_website:
                        continue
                    restart = False
                runconfigs.append({"scenarioname":scenario["name"], "website":website})

    for runconfig in runconfigs:
        if True:
            scenario = scenario_by_name[runconfig["scenarioname"]]
            website = runconfig["website"]
            logger.info(f"Training in scenario {scenario} on {website}")
            config["website"] = website
            config["scenario"] = scenario["name"]
            config["bwdown"] = scenario["bwdown"]
            config["rtt"] = scenario["rtt"]
            config["loss"] = scenario["loss"]
            if Debug:
                logger.warning("Warning: Using Debug Settings!")
                config["n_iter"] = 1
                config["reps"] = 1
                config["ch_reps"] = 1
                config["eval_reps"] = 1
            try:
                train_website(config)
                logger.info(f"Training {website} done")
            except Exception as e:
                logger.error(f"Error in training {website}: {e}")
                continue
