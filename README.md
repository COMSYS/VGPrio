# Visually Guided HTTP/3 Prioritization Learning Framework
This repository contains the learning framework implementation for our paper *Visually Guided HTTP/3 Prioritization*.

## Publication

* Constantin Sander, Ike Kunze, Dario Veltri and Klaus Wehrle: *VGPrio: Visually Guided HTTP/3 Prioritization*. In Proceedings of the IFIP Networking Conference (TMA '25), 2025.

If you use any portion of our work, please consider citing our publication.

```
@inproceedings {2025-sander-vgprio,
   title = {{VGPrio: Visually Guided HTTP/3 Prioritization}},
   year = {2025},
   booktitle = {Proceedings of the IFIP Networking Conference (Networking '25)},
   author = {Sander, Constantin and Kunze, Ike and Veltri, Dario and Wehrle, Klaus}
}
```

## Contents
The repository consists of the learning module and the testbed, which is loaded as a subrepository, from our previous HTTP Prioritization paper.

### Prerequisites
To run the learning framework, our testbed needs to be built first. Thus, please follow the instructions in the testbed repo first for setup and generation of priohints.
Thereafter, the actual learning framework can be used.
Here, we assume that mahimahi files of all pages reside under ```testbed/mm/record/<page>/*.save``` and priohints reside under ```testbed/mm/priorities/<page>.csv```.

### Training
For the training, ```python3 learning/training.py``` starts training on all available websites with scenarios defined in ```learning/scenarios.json```.
```learning/conf.json``` defines the general configuration for training, i.e. overall iterations or number of repetitions per iteration.

### Evaluation
After every scenario was fully trained, an evaluation run is performed against the Firefox, RR and Chrome strategy.
Evaluation results are saved under ```learning/run_data/<scenario>/<page>```.

### Dataset
```results.csv``` contains our webperformance results per run
