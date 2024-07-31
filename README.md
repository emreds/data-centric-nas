# Data Centric NAS Initialization
This repository contains the code for the method of data centric initialization of Neural Architecture Search. 

## Installation 
- Create a virtual environment by `python -m venv .venv`
- Activate the virtual environment based on your operating system.
- `pip install -r requirements.txt`
- To install the naslib refer to -> https://github.com/automl/NASLib

## Usage 
- Define the `STARTING_POINTS` and `surr_train_cnt` and `raw_mo_train_cnt` in `main.py`. 
- Run it by running the command `python main.py`.
- To run the data centric initialization, it needs surrogate models trained on NB101 search space and they need to be placed under the folder `surrogates`.

## How it works? 
- For every given surrogate model, we make the performance prediction of every architecture based on validation accuracy and train time in NB101 and pick the top 20 on the pareto front of it.
- Then, we explore the neighborhood of those architectures in predicted pareto front and update the pareto front. 
- After that, instead of making predictions and picking architectures from pareto-front, we randomly pick architectures from the NB101.
- We explore the neighborhood of those randomly picked architectures and add the best performing ones to pareto front.
- In the end, we compare the surroage initialized exploration round with the randomly initialized exploration round. 
