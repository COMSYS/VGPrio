# Run Data
All run data is stored in the `run_data` directory contained in the `learning` directory. The Run data is structured as follows:

At the top level, log files showing information about the current training, the current learning run and it's processes (for debugging).
Folders at root level contain the run data associated with a specific network scenario.

## Data of specific Runs
In a scenario folder, training runs are organized by website. For each website, the following data is stored by the learning framework:
 *  For each iteration, the testbed data is stored in folder named after the iteration number containing for each redundant rep: lists of HTTP requests and measurement results and the priorities used as CSV
 *  evaluation measurements for the modes `priofile`, `chrome` and `firefox` are stored in equally named folders
 *  the training configuration in `config.json`
 *  training and testbed logs
 *  plots showing the image clustering and all iteration speeds (`img_clustering.png` and `iterations.png`)
 *  as well as all gathered training data once the run is complete, contained in `run_data.json`