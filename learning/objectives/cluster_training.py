import json
import time
from datetime import timedelta

import numpy as np
from bayes_opt import BayesianOptimization, Events, JSONLogger

from learning.config import Config
from learning.run import Run
from learning.utils import prio_switch


def cluster_training(
    website: str,
    scenario: str,
    uid: int,
    gid: int,
    init_points=1,
    n_iter=300,
    ch_reps=30,
    mean=False,
    repeat=10,
    run_name="training",
    log_tb=False,
    incremental=False,
    bwdown=10,
    rtt=100,
    bdp=1,
    loss=0.0,
):
    """
    Main function for training a model using Bayesian optimization for optimizing a websites speed index

    Args:
        website (str): The website to optimize
        scenario (str): The scenario to optimize
        uid (int): The user id to use for the test
        gid (int): The group id to use for the test
        init_points (int, optional): Number of initial points to sample. Defaults to 1.
        n_iter (int, optional): Number of iterations to run. Defaults to 300.
        ch_reps (int, optional): Number of repetitions to run Chrome for noise level. Defaults to 30.
        mean (bool, optional): Use mean instead of median for SI. Defaults to False.
        repeat (int, optional): Number of repetitions in each iteration. Defaults to 10.
        run_name (str, optional): Name of the run. Defaults to "training".
        log_tb (bool, optional): Log testbed. Defaults to False.
        incremental (bool, optional): Use incremental mode. Defaults to False (deprecated).
        bwdown (int, optional): Bandwidth in Mbit/s. Defaults to 10.
        rtt (int, optional): Round trip time in ms. Defaults to 100.
        bdp (int, optional): Bandwidth delay product. Defaults to 1.
        loss (float, optional): Packet loss in %. Defaults to 0.0.

    Returns:
        Run: The run object
    """

    def cluster_objective(**kwargs):
        """
        Objective function for the Bayesian optimization.
        """
        start_time = time.time()
        run.logger.info(f"Parameters: {kwargs}")

        run.priorities.loc[run.priorities["type"] == "css", "priority"] = prio_switch(1)

        run.priorities.loc[
            (run.priorities["type"] == "js")
            & (run.priorities["async"] == 0)
            & (run.priorities["defer"] == 0),
            "priority",
        ] = prio_switch(1)

        prios = []
        img_prios = len([k for k, v in kwargs.items() if "img" in k])
        if incremental:
            img_prios = img_prios // 2
        for i in range(img_prios):
            prio = prio_switch(round(kwargs[f"img_{i}"]))
            prios.append(prio)
            run.priorities.loc[
                (run.priorities["type"] == "image") & (run.priorities["cluster"] == i),
                "priority",
            ] = prio
            if incremental:
                inc_prio = round(kwargs[f"img_{i}_inc"])
                prios.append(inc_prio)
                run.priorities.loc[
                    (run.priorities["type"] == "image")
                    & (run.priorities["cluster"] == i),
                    "incremental",
                ] = inc_prio

        js_async_prio = prio_switch(round(kwargs["js_async"]))
        prios.append(js_async_prio)
        run.priorities.loc[
            (run.priorities["type"] == "js")
            & ((run.priorities["async"] == 1) | (run.priorities["defer"] == 1)),
            "priority",
        ] = js_async_prio
        if incremental:
            js_async_inc_prio = round(kwargs["js_async_inc"])
            prios.append(js_async_inc_prio)
            run.priorities.loc[
                (run.priorities["type"] == "js")
                & ((run.priorities["async"] == 1) | (run.priorities["defer"] == 1)),
                "incremental",
            ] = js_async_inc_prio
            # set inc for rest
            inc_rest = round(kwargs["inc_rest"])
            prios.append(inc_rest)
            run.priorities.loc[
                (~(run.priorities["type"] == "image"))
                & (
                    ~(
                        (run.priorities["type"] == "js")
                        & (
                            (run.priorities["async"] == 1)
                            | (run.priorities["defer"] == 1)
                        )
                    )
                ),
                "incremental",
            ] = inc_rest

        run.logger.info(f"Priorities: {prios}")

        si = 0
        while si == 0:
            run.iterate(mode="iteration", incremental=incremental)
            si, stddev = run.get_si(mode="iteration", mean=mean)
        run.log_prios(iteration=run.iteration)
        run_data["iterations"].append(
            {
                "iteration": run.iteration,
                "si": si,
                "stddev": stddev,
                "parameters": kwargs,
                "prios": prios,
            }
        )
        if si < run.best_si:
            run.best_si = si
            run.best_iteration = run.run_dir / str(run.iteration)
        with open(run.run_dir / "run_data.json", "w") as f:
            json.dump(run_data, f, indent=4)
        end_time = time.time()
        iteration_time = end_time - start_time
        iterations_left = n_iter - run.iteration
        expected_time = timedelta(seconds=iteration_time * iterations_left)
        if iterations_left > 0:
            run.logger.info(f"Expected time left: {expected_time}")
        return -si

    config = Config(
        website,
        scenario,
        uid,
        gid,
        run_name=run_name,
        repeat=repeat,
        bwdown=bwdown,
        rtt=rtt,
        bdp=bdp,
        loss=loss,
    )
    run = Run(config, log_testbed=log_tb)

    parameters = []
    for i in range(run.priorities["cluster"].nunique(dropna=True)):
        parameters.append(
            {
                "name": f"img_{i}",
                "domain": (0.51, 5.49),
            }
        )
        if incremental:
            parameters.append(
                {
                    "name": f"img_{i}_inc",
                    "domain": (0.0, 1.0),
                }
            )
    parameters.append(
        {
            "name": f"js_async",
            "domain": (0.51, 5.49),
        }
    )
    if incremental:
        parameters.append(
            {
                "name": f"js_async_inc",
                "domain": (0.0, 1.0),
            }
        )
        parameters.append(
            {
                "name": "inc_rest",
                "domain": (0.0, 1.0),
            }
        )

    param_bounds = {param["name"]: param["domain"] for param in parameters}

    # find noise level
    run.logger.info(
        f"Finding noise level. Running Chrome prioritization for {ch_reps} reps"
    )
    run.iterate(mode="noise", reps=ch_reps, incremental=incremental)
    si, stddev = run.get_si(mode="noise", reps=ch_reps)
    alpha = (stddev / si) ** 2

    run_data = {}

    run_data["Chrome-SpeedIndex"] = si
    run.logger.info(f"Chrome-SpeedIndex: {si}, stddev: {stddev}")

    seed = np.random.randint(0, 100000)
    run.logger.info(f"Seed: {seed}")
    run_data["seed"] = seed
    run_data["iterations"] = []

    optimizer = BayesianOptimization(
        f=cluster_objective,
        pbounds=param_bounds,
        verbose=2,
        random_state=seed,
        allow_duplicate_points=True,
    )

    logger = JSONLogger(path=str(run.run_dir / "optimizer_logs.json"))
    optimizer.subscribe(event=Events.OPTIMIZATION_STEP, subscriber=logger)

    optimizer.set_gp_params(alpha=alpha)
    run.logger.info(f"Set alpha to: {alpha}")

    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    run.logger.info(f"Best parameters: {optimizer.max['params']}")
    run.logger.info(f"Best target: {optimizer.max['target']}")

    run_data["best_iteration"] = str(run.best_iteration)
    run_data["best_params"] = optimizer.max["params"]
    run_data["best_target"] = optimizer.max["target"]

    with open(run.run_dir / "run_data.json", "w") as f:
        json.dump(run_data, f, indent=4)
    return run
