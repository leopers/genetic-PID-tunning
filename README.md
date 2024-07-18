# Evolutionary Strategies for PID Controller Tuning

## About

This project aims to compare the performance of two distinct methods for tuning PID controllers: the Genetic Algorithm (GA) and Ziegler-Nichols' heuristics.

PID controllers are widely used in industrial applications due to their simplicity and effectiveness in controlling dynamic systems. Traditional methods like Ziegler-Nichols provide quick initial tuning based on empirical rules, but often require further adjustments. On the other hand, computational optimization techniques such as GA explore the search space more comprehensively, potentially yielding better performance in complex and nonlinear systems.

This study involves implementing each tuning method, running simulations, and evaluating the results based on criteria such as response time, overshoot, and steady-state error. The goal is to provide a detailed comparison that highlights the strengths and weaknesses of each approach, contributing to more informed decision-making in industrial control applications.

## Requirements

- python
- numpy
- tqdm
- matplotlib
- control

## Quick Start

1. Clone the repo

   ```sh
   git clone https://github.com/leopers/genetic-PID-tunning
   ```

2. Install the dependencies on repo directory

   ```sh
   pip install -r requirements.txt
   ```

3. Customize experiment parameters on main.py (optional)

4. Run the project:
   ```sh
   cd src
   python main.py
   ```

## Follow us

- https://github.com/andre-thiessen
- https://github.com/leopers
- https://github.com/jrsHenrique
