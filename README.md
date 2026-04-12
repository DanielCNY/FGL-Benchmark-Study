# FGL Benchmark Study

Code and results for my Final Year Project on benchmarking Federated Graph Learning (FGL) under data heterogeneity.

## Overview

This repository contains the benchmark code, the raw outputs from individual simulations, and the script used to aggregate those outputs into the final reported results.

The benchmark compares FedAvg, FedProx, FedNova, and a lightweight environment-aware prototype across homophilic and heterophilic graph datasets under a shared experimental setting.

## Repository Structure

pyproject.toml
Defines the settings for a single simulation, such as the algorithm, dataset, number of rounds, and number of clients.

run_test.py
Runs one single simulation using the settings in pyproject.toml.

analysis/
Stores the outputs from many completed simulations. These files are organised into subfolders, for example by seed or run group.

aggregate_results.py
Reads all valid simulation summaries inside analysis/ and produces the aggregated results used for reporting.

## How the workflow works

This repository is organised around two stages.

### 1. Run a single simulation

A single benchmark run is configured through pyproject.toml.

For example, this is where you set:

- the algorithm
- the dataset
- the number of communication rounds
- the number of clients
- any other run-specific settings

Once those settings are chosen, run:

python run_test.py

This produces the output for one simulation only.

### 2. Build up a collection of simulations

The full benchmark results in the dissertation were not produced from one run. They were produced from many separate simulations, each configured and run individually.

After each run, the resulting summary files were placed into the analysis/ folder and organised into its current subfolder structure.

Any new runs must be added in the same way and must follow the existing naming and folder conventions if they are to be included in the final aggregation.

### 3. Aggregate the full results

Once the analysis/ folder contains all of the simulations you want to include, run:

python aggregate_results.py

This script does not run new experiments. It only reads the simulation summaries already present in analysis/ and produces the aggregated results.

## Adding new runs

To add more results to the benchmark:

1. update the settings in pyproject.toml
2. run one simulation with python run_test.py
3. place the resulting summary files into the appropriate location inside analysis/
4. keep the existing naming and folder structure consistent
5. rerun python aggregate_results.py

If the new files match the existing format and organisation, they will be included automatically in the aggregated outputs.

## Reproducibility

This repository includes:

- the code used to run individual simulations
- the configuration used to control those simulations
- the collected raw summaries from many runs
- the aggregation script used to produce the final benchmark results

## Project Context

This repository supports the dissertation:

Benchmarking Federated Graph Learning Methods under Data Heterogeneity

The goal of the project is to compare federated graph learning methods under shared heterogeneous conditions and to evaluate whether a lightweight environment-aware aggregation strategy can improve robustness across structurally different graph settings.