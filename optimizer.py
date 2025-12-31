import random
import logging
import copy
from config_loader import ConfigLoader
from experiment_runner import ExperimentRunner

logger = logging.getLogger("GA_Optimizer")


class GeneticOptimizer:
    """Genetic Algorithm for PID Tuning"""

    def __init__(self, cfg_path, pop_size=10, generations=5):
        self.base_cfg = ConfigLoader.load(cfg_path)
        self.pop_size = pop_size
        self.generations = generations
        self.bounds = {'kp': [500, 5000], 'ki': [0, 50], 'kd': [0, 1000]}

    def fitness(self, params):
        # Run simplified simulation to evaluate parameters
        test_cfg = copy.deepcopy(self.base_cfg)
        test_cfg['vehicle_types']['Heavy_Hauler']['pid'] = params
        test_cfg['simulation']['duration'] = 50.0  # Short run
        test_cfg['simulation']['visualization'] = False

        runner = ExperimentRunner(test_cfg)
        runner.run()

        # Calculate Cost: ISE (Integral Square Error) + Energy
        err = sum([(l['vel'] - 6.0) ** 2 for l in runner.logs if l['state'] == 'MOVING'])
        energy = sum([l['energy'] for l in runner.logs])
        return err + 0.001 * energy

    def run(self):
        pop = [{k: random.uniform(*v) for k, v in self.bounds.items()} for _ in range(self.pop_size)]

        for g in range(self.generations):
            scores = [(ind, self.fitness(ind)) for ind in pop]
            scores.sort(key=lambda x: x[1])
            logger.info(f"Gen {g}: Best Cost {scores[0][1]:.2f} Params {scores[0][0]}")

            # Selection & Crossover
            next_pop = [s[0] for s in scores[:2]]  # Elitism
            while len(next_pop) < self.pop_size:
                p1, p2 = random.choice(scores[:5])[0], random.choice(scores[:5])[0]
                child = {k: (p1[k] + p2[k]) / 2 * random.uniform(0.9, 1.1) for k in p1}
                next_pop.append(child)
            pop = next_pop