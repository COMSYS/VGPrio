import csv
import datetime
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from sklearn.cluster import KMeans

from learning.config import Config
from learning.utils import change_dir_mod, prio_switch, read_priofile, write_priofile
from testbed.testbed import Testbed


class Run:
    """Encapsulates run functionality. Keeps track of iteration directories and priorities."""

    def __init__(self, config: Config, log_testbed=True, priofile=None):
        self.name = config.run_name
        self.namespace = config.namespace
        self.config = config

        self.run_dir = config.run_dir
        self.evalconfig = config.evalconfig
        self.website = self.config.website

        self.iteration = 0
        self.best_iteration = 0
        self.best_si = sys.maxsize
        self.run_dir.mkdir(parents=True, exist_ok=True)
        change_dir_mod(self.run_dir, self.config.user_id, self.config.group_id)

        self.logger = logging.getLogger(self.namespace + "-run")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
        handler = logging.FileHandler(self.run_dir / "run.log")
        handler.setLevel(logging.INFO)
        handler_current = logging.FileHandler(
            self.config.run_data_dir / "current_run.log"
        )
        with open(self.config.run_data_dir / "current_run.log", "w"):
            pass  # clears log file
        handler_current.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.ERROR)
        formatter = logging.Formatter(
            "[%(asctime)s-%(name)s-%(levelname)s]: %(message)s"
        )
        handler.setFormatter(formatter)
        handler_current.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        self.logger.addHandler(handler)
        self.logger.addHandler(handler_current)

        self.testbed = Testbed(
            evalconfig=self.evalconfig,
            namespace=self.namespace,
            run_dir=self.run_dir,
            run_logger=self.logger,
        )

        if not priofile:
            self.priorities = read_priofile(config.website_priorities)
        else:
            self.priorities = read_priofile(priofile)
        self.priorities.columns = [
            "src",
            "type",
            "priority",
            "ffclass",
            "ffweight",
        ]

        self.logger.info("Obtaining list of resources using Browsertime...")
        self.iterate(mode="browsertime")
        data = self.get_script_info()
        with open(self.run_dir / "data.json", "w") as f:
            json.dump(data, f, indent=4)
        new_resources = pd.DataFrame(data)
        new_resources = new_resources[new_resources["src"].str.contains("https://")]
        new_resources["src"] = new_resources["src"].str.replace(
            "https://", "", regex=False
        )
        self.priorities["src"] = self.priorities["src"].str.replace(
            "GET:", "", regex=False
        )
        self.priorities = self.priorities.merge(new_resources, on="src", how="outer")
        self.priorities["type_x"] = self.priorities["type_x"].fillna(
            self.priorities["type_y"]
        )
        self.priorities = self.priorities.rename(columns={"type_x": "type"})
        self.priorities = self.priorities.drop("type_y", axis=1)

        self.priorities["priority"] = prio_switch(3)  # initialize with default value
        self.priorities["incremental"] = 0  # not contained in initial priofile
        self.priorities["ffclass"] = self.priorities["ffclass"].fillna("followers")
        self.priorities["ffweight"] = self.priorities["ffweight"].fillna(22)

        # avoid any # characters
        self.priorities["src"] = self.priorities["src"].str.split("#").str[0]

        # for websites that do not contain external js
        if not (self.priorities["type"] == "js").any():
            self.priorities["async"] = pd.NA
            self.priorities["defer"] = pd.NA

        self.log_prios()

        self.resources = [
            str(resource).replace("GET:", "")
            for resource in self.priorities["src"].to_list()
        ]

        self.cluster_images()

        self.save_prios()

        if log_testbed:
            self.testbed.logger.addHandler(handler)
            self.testbed.logger.addHandler(handler_current)
        self.logger.info("Run initialized")

    def log_prios(self, iteration=None):
        """Log the complete priorties dataframe to csv."""
        if iteration:
            iteration_path = self.run_dir / str(iteration)
            self.priorities.to_csv(iteration_path / "prio.csv", sep="#")
        else:
            self.priorities.to_csv(self.run_dir / "prio.csv", sep="#")

    def process_log_to_csv(self, rep_path: Path):
        """
        Reads requests dumped to error.log files from multiple h2o servers and converts them to CSV

        Args:
            rep_path: Path object for the report directory
            delimiter: CSV delimiter character (default: comma)
        """
        try:
            for i, _ in enumerate(self.testbed.servers):
                server_dir = rep_path / f"server{i}"
                if server_dir.is_dir():
                    input_file = server_dir / "error.log"
                    output_file = rep_path / f"requests{i}.csv"
                    if not input_file.stat().st_size == 0:
                        with open(input_file, "r") as file:
                            for _ in range(2):  # Skip first two lines
                                next(file)

                            content_lines = []
                            for line in file:
                                if line.strip():
                                    # Split the line and create a dict with proper headers
                                    values = next(
                                        csv.reader([line.strip()], delimiter="#")
                                    )
                                    row_dict = dict(
                                        zip(
                                            [
                                                "src",
                                                "chrome class",
                                                "urgency",
                                                "incremental",
                                            ],
                                            values,
                                        )
                                    )
                                    content_lines.append(row_dict)

                            if not content_lines:
                                continue

                            with open(output_file, "w", newline="") as csvfile:
                                writer = csv.DictWriter(
                                    csvfile,
                                    fieldnames=[
                                        "src",
                                        "chrome class",
                                        "urgency",
                                        "incremental",
                                    ],
                                    delimiter="#",
                                    quoting=csv.QUOTE_MINIMAL,
                                )
                                writer.writeheader()
                                writer.writerows(content_lines)

                            self.logger.info(
                                f"Successfully processed server{i}/error.log to requests{i}.csv"
                            )

        except FileNotFoundError as e:
            self.logger.error(f"Error: Could not find file: {e}")
        except csv.Error as e:
            self.logger.error(f"Error parsing CSV: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")

    def extract_urls_from_access_log(self, rep_path: Path):
        """
        Reads a log file and extracts full URLs by combining the host and route from HTTP requests.

        Args:
            log_file_path (str or Path): Path to the log file

        Returns:
            list: List of complete URLs
        """
        urls = []
        http_methods = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}

        try:
            for i, _ in enumerate(self.testbed.servers):
                server_dir = rep_path / f"server{i}"
                if server_dir.is_dir():
                    with open(server_dir / "access.log", "r") as file:
                        for line in file:
                            parts = line.strip().split()
                            if len(parts) < 8:
                                continue

                            for i1, part in enumerate(parts):
                                part = part.replace('"', "")
                                if part in http_methods:
                                    route = parts[i1 + 1]
                                    host = parts[-1]  # Host is the last element

                                    route = route.strip('"')
                                    host = host.strip('"')

                                    # Combine host and route
                                    if route.startswith("http"):
                                        urls.append(route)
                                    else:
                                        full_url = f"https://{host}{route}"
                                        urls.append(full_url)
                                    break

                    if urls:
                        with open(rep_path / "access.json", "w", newline="") as f:
                            json.dump(urls, f)
                    else:
                        self.logger.info("No URLs found in log file")

                    requests_csv_path = rep_path / f"requests{i}.csv"
                    if requests_csv_path.exists():
                        requests = pd.read_csv(requests_csv_path, sep="#")
                        request_src = requests["src"].str.replace("GET:", "")
                        urls = [str(u).replace("https://", "") for u in urls]
                        unique_to_urls = list(set(urls) - set(request_src))
                        with open(rep_path / "unrequested.json", "w") as f:
                            json.dump(unique_to_urls, f)

        except FileNotFoundError as e:
            self.logger.error(f"Error: Could not find file: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error processing log file: {e}")
            return False

    def cluster_images(self):
        """Cluster all images that have the necessary information associated with them"""
        cluster_images = self.priorities[
            (self.priorities["type"] == "image")
            & self.priorities[["isAboveTheFold", "height", "width"]].notna().all(axis=1)
        ].copy()

        cluster_images["width"] = cluster_images["width"].astype(float)
        cluster_images["height"] = cluster_images["height"].astype(float)
        cluster_images["isAboveTheFold"] = cluster_images["isAboveTheFold"].astype(int)
        cluster_images[
            "isAboveTheFold"
        ] *= 10000  # scale to increase importance in clustering
        kmeans = KMeans(n_clusters=3, init="k-means++")
        clustering = kmeans.fit(cluster_images[["width", "height", "isAboveTheFold"]])
        cluster_images["cluster"] = clustering.labels_
        self.plot_img_clustering(cluster_images, clustering.cluster_centers_)
        self.priorities = self.priorities.merge(
            cluster_images[["src", "cluster"]], on="src", how="left"
        )

    def update_prios(self, new_prios: dict):
        """Update the priorities of the resources."""
        try:
            for resource, prio in new_prios.items():
                new_prio = prio_switch(round(prio))
                self.priorities.loc[self.priorities["src"] == resource, "priority"] = (
                    new_prio
                )

        except Exception as e:
            self.logger.error(f"Error updating priorities: {e}")

    def save_prios(self, iteration=""):
        """Save the priorities to a file."""
        try:
            write_priofile(
                self.run_dir / str(iteration) / "priorities.csv", self.priorities
            )
            change_dir_mod(self.run_dir, self.config.user_id, self.config.group_id)
        except Exception as e:
            self.logger.error(f"Error saving priorities: {e}")

    def iterate(self, mode="", reps=0, priofile="", incremental=False):
        """Run the testbed and save the results."""
        try:
            if mode not in [
                "browsertime",
                "chrome",
                "firefox",
                "rr",
                "priofile",
                "iteration",
                "noise",
            ]:
                raise ValueError(f"Invalid mode {mode}")
            iter_path = self.run_dir
            if mode == "browsertime":
                start = time.time()
                self.testbed.setup(lighthouse=False)
                iter_path = iter_path / "browsertime"
                iter_path.mkdir(parents=True, exist_ok=True)
                self.testbed.run_browsertime(iter_path)
                self.testbed.cleanup(lighthouse=False)
                end = time.time()
                self.logger.info(f"Browsertime took {end - start} seconds")
                return
            elif mode == "iteration":
                reps = self.evalconfig["repeat"]
                self.iteration += 1
                iter_path = iter_path / str(self.iteration)
                iter_path.mkdir(parents=True, exist_ok=True)
                self.save_prios(self.iteration)
                self.testbed.evalconfig["priomode"] = "chromeext"
                if incremental:
                    self.testbed.evalconfig["priomode"] = "chromeextinc"
                self.testbed.evalconfig["priorities"] = str(
                    self.run_dir / str(self.iteration) / "priorities.csv"
                )
            else:
                iter_path = iter_path / mode
                self.testbed.evalconfig["priorities"] = str(
                    self.run_dir / "priorities.csv"
                )
                if mode in ["chrome", "priofile", "noise"]:
                    self.testbed.evalconfig["priomode"] = "chromeext"
                    if mode == "priofile":
                        if incremental:
                            self.testbed.evalconfig["priomode"] = "chromeextinc"
                    if mode == "priofile":
                        if priofile:
                            self.testbed.evalconfig["priorities"] = priofile
                        else:
                            self.testbed.evalconfig["priorities"] = str(
                                self.run_dir
                                / str(self.best_iteration)
                                / "priorities.csv"
                            )
                else:
                    if mode == "rr":
                        self.testbed.evalconfig["priomode"] = "rrext"
                    else:
                        self.testbed.evalconfig["priomode"] = "firefoxext"
            start = time.time()
            for rep in range(1, reps + 1):
                rep_start = time.time()
                rep_path = iter_path / f"rep{rep}"
                self.testbed.setup(lighthouse=True)
                rep_path.mkdir(parents=True, exist_ok=True)
                with open(rep_path / "evalconfig.json", "w") as f:
                    json.dump(self.testbed.evalconfig, f, indent=4)
                self.testbed.run_lighthouse(rep_path)
                self.process_log_to_csv(rep_path=rep_path)
                self.extract_urls_from_access_log(rep_path=rep_path)
                self.testbed.cleanup(lighthouse=True)
                # delete netlog files for iterations to save space
                if (mode in ["iteration", "noise"]) or (rep != 1):
                    netlog = rep_path / "lighthouse" / "netlog.json"
                    if netlog.exists():
                        netlog.unlink()
                rep_end = time.time()
                self.logger.info(f"Rep {rep} took {rep_end - rep_start} seconds")
                time.sleep(0.1)
            end = time.time()
            self.logger.debug(f"Iteration {self.iteration} took {end - start} seconds")
        except Exception as e:
            self.logger.error(f"Error iterating: {e}")

    def get_si(self, mode="", reps=0, mean=True) -> float:
        """Get the Speed Index from the last iteration."""
        try:
            iter_path = self.run_dir / str(self.iteration)
            speeds = []
            dump_data = {}
            dump_data["reps"] = []

            if mode in ["chrome", "noise", "firefox", "priofile", "rr"]:
                iter_path = self.run_dir / mode

            elif mode == "iteration":
                iter_path = self.run_dir / str(self.iteration)
                reps = self.evalconfig["repeat"]

            for rep in range(1, reps + 1):
                rep_path = iter_path / f"rep{rep}"
                with open(rep_path / "lighthouse" / "lighthouse-report.json", "r") as f:
                    results = json.load(f)
                speeds.append(
                    results["audits"]["metrics"]["details"]["items"][0][
                        "observedSpeedIndex"
                    ]
                )
                dump_data["reps"].append({"rep": rep, "si": speeds[-1]})

            stdev = np.std(speeds)
            mean = np.mean(speeds)
            median = np.median(speeds)
            dump_data["mean"] = mean
            dump_data["median"] = median
            dump_data["stdev"] = stdev

            with open(iter_path / "iteration.json", "w") as f:
                json.dump(dump_data, f)

            if mode == "iteration":
                self.logger.info(
                    f"Iteration {self.iteration} ended with: si_mean: {round(mean, 2)}, si_median: {round(median, 2)},  stdev: {round(stdev, 2)}"
                )
            else:
                self.logger.info(
                    f"Mode {mode} ended with: si_mean: {round(mean, 2)}, si_median: {round(median, 2)},  stdev: {round(stdev, 2)}"
                )
            if mean:
                return mean, stdev
            else:
                return median, stdev
        except Exception as e:
            self.logger.error(f"Error getting SI: {e}")
            return 0, 0

    def run_static_mode(self, mode="", reps=0, best_iteration=0, priofile=""):
        """Run a static mode with a given number of repetitions."""
        try:
            if mode not in ["chrome", "firefox", "priofile", "rr"]:
                raise ValueError("Invalid mode {mode}")
            if reps == 0:
                raise ValueError("Invalid number of repetitions")
            if mode == "priofile":
                if self.best_iteration == 0:
                    if best_iteration == 0:
                        raise ValueError(
                            "No best iteration found, run optimization first"
                        )
                    else:
                        self.best_iteration = best_iteration
            self.logger.info(f"Running static mode: {mode} for {reps} reps")
            self.iterate(mode=mode, reps=reps, priofile=priofile, incremental=True)
            si, stddev = self.get_si(mode=mode, reps=reps)
            return si, stddev
        except Exception as e:
            self.logger.error(f"Error running static mode: {e}")
            return 0, 0

    def get_script_info(self) -> list[dict]:
        """Extract information about resources via javascript run on browsertime."""
        try:
            with open(
                self.run_dir / "browsertime/browsertime/browsertime.json",
                "r",
            ) as f:
                data = json.load(f)
            return data[0]["browserScripts"][0]["custom"]["script"]
        except Exception as e:
            self.logger.error(f"Error getting script info: {e}")
            return None

    def calculate_traces(self, path: Path):
        """Calculate the difference in order of requested and served files."""
        with open(path / "browsertime/browsertime.har", "r") as f:
            data = json.load(f)

        entries = data["log"]["entries"]
        for entry in entries:
            timestamp_s = entry["startedDateTime"].replace("Z", "+00:00")
            entry["req"] = datetime.datetime.strptime(
                timestamp_s, "%Y-%m-%dT%H:%M:%S.%f%z"
            )
            timings = entry["timings"]
            entry["received"] = entry["req"] + datetime.timedelta(
                milliseconds=timings["send"] + timings["wait"] + timings["receive"]
            )
        req = [{"url": e["request"]["url"], "t": e["req"]} for e in entries]
        ser = [{"url": e["request"]["url"], "t": e["received"]} for e in entries]

        req.sort(key=lambda x: x["t"])
        ser.sort(key=lambda x: x["t"])

        req_r = [e["url"] for e in req]
        ser_r = [e["url"] for e in ser]

        # Create a DataFrame with a column for each list
        df = pd.DataFrame(
            {
                "rec": [e["request"]["url"].replace("https://", "") for e in entries],
                "req": req_r,
                "req_order": [req_r.index(e["request"]["url"]) for e in entries],
                "ser": ser_r,
                "ser_order": [ser_r.index(e["request"]["url"]) for e in entries],
            }
        )

        # Calculate the difference in order
        df["order_diff"] = df["ser_order"] - df["req_order"]
        df["priority"] = ""
        df["type"] = ""
        df["cluster"] = 0

        for i, r in self.priorities.iterrows():
            df.loc[df["rec"] == r["resource"], ["priority", "type", "cluster"]] = (
                r["priority"],
                r["type"],
                r["cluster"],
            )

        df[
            [
                "rec",
                "type",
                "priority",
                "cluster",
                "req_order",
                "ser_order",
                "order_diff",
            ]
        ].to_csv(path / "trace.csv", index=False)

    def plot_iterations(self):
        """Create scatterplot of si values in all iterations"""
        with open(self.run_dir / "run_data.json", "r") as f:
            run_data = json.load(f)
        iterations = run_data["iterations"]
        plt.figure(figsize=(10, 6))
        iterations_x = [iteration["iteration"] for iteration in iterations]
        si_values = [iteration["si"] for iteration in iterations]
        stddev_values = [iteration["stddev"] for iteration in iterations]
        plt.scatter(iterations_x, si_values, label="SI")
        plt.fill_between(
            iterations_x,
            [si - stddev for si, stddev in zip(si_values, stddev_values)],
            [si + stddev for si, stddev in zip(si_values, stddev_values)],
            color="b",
            alpha=0.1,
        )

        plt.title("Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("SI")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.run_dir / "iterations.png")
        plt.clf()

    def plot_mode_speeds(self):
        """Create scatterplots for speeds in all modes for comparison"""
        plt.figure(figsize=(10, 6))
        modes = ["chrome", "firefox", "priofile", "rr"]
        speeds = {"chrome": [], "firefox": [], "priofile": [], "rr":[]}
        for mode in modes:
            with open(self.run_dir / f"{mode}" / "iteration.json", "r") as f:
                iterations = json.load(f)["reps"]
                for iter in iterations:
                    speeds[mode].append(iter["si"])
        for mode in modes:
            plt.scatter([mode] * len(speeds[mode]), speeds[mode], label=mode)

        plt.title(f"{self.website} Speeds")
        plt.xlabel("Mode")
        plt.ylabel("Speed")
        plt.grid(True)
        plt.savefig(self.run_dir / "speeds.png")
        plt.clf()

    def plot_img_clustering(self, df_images: pd.DataFrame, img_centers):
        """Plot clustering of images"""
        num_clusters = len(img_centers)
        cluster_palette = sns.color_palette("hsv", num_clusters)
        df_images["cluster_color"] = df_images["cluster"].map(
            lambda x: cluster_palette[x]
        )
        center_colors = [cluster_palette[i] for i in range(num_clusters)]

        plt.figure(figsize=(10, 6))

        for fold_status, marker in zip(
            [10000, 0], ["o", "^"]
        ):  # o for above, ^ for below
            subset = df_images[df_images["isAboveTheFold"] == fold_status]
            plt.scatter(
                subset["width"],
                subset["height"],
                c=subset["cluster_color"],
                marker=marker,
                label=f"Above the Fold: {fold_status}",
                alpha=0.6,
            )

        plt.scatter(
            img_centers[:, 0],
            img_centers[:, 1],
            c=center_colors,
            s=100,
            alpha=0.75,
            marker="X",
            label="Cluster Centers",
        )

        plt.title("Image Attributes with KMeans Clustering")
        plt.xlabel("Width")
        plt.ylabel("Height")

        legend_elements = [
            Patch(facecolor=cluster_palette[i], label=f"Cluster {i+1}")
            for i in range(num_clusters)
        ]
        legend_elements.append(
            Patch(facecolor="none", edgecolor="none", label="Markers:")
        )
        legend_elements.append(
            Patch(facecolor="none", edgecolor="w", label="X: Cluster Centers")
        )
        legend_elements.append(
            Patch(facecolor="none", edgecolor="none", label="o: Above the Fold")
        )
        legend_elements.append(
            Patch(facecolor="none", edgecolor="none", label="^: Below the Fold")
        )

        plt.legend(
            handles=legend_elements,
            title="Legend",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.run_dir / "img_clustering.png")
        plt.clf()
