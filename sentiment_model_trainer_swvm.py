from sentiment_model_trainer_base import SentimentModelTrainerBase

import numpy as np
import cvxpy as cvx


class SentimentModelTrainerSWVM(SentimentModelTrainerBase):

    def weights_update_step_on_batch(self, w, documents_batch: list, feature_vectors_batch: list,
                                     inferred_labelings_batch: list):
        """Not implemented yet."""
        pass
        return None

