from abc import ABC, abstractmethod
import numpy as np


class OptimizerBase(ABC):
    @abstractmethod
    def find_next_weights_according_to_constraints(self, previous_w: np.ndarray,
                                                   G: np.ndarray, L: np.ndarray):
        """ G*w >= L """
        ...


class OptimizerLoader:
    @staticmethod
    def load_optimizer(optimizer_name: str):
        SUPPORTED_OPTIMIZERS = {'qp', 'cvx'}

        if optimizer_name == 'qp':
            from qp_optimizer import QPOptimizer
            return QPOptimizer()
        elif optimizer_name == 'cvx':
            from cvx_optimizer import CVXOptimizer
            return CVXOptimizer()

        raise ValueError('Unknown optimizer `{}`. Supported optimizers: {}'.format(
            optimizer_name, SUPPORTED_OPTIMIZERS))
