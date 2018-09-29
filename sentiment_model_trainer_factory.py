from sentiment_model_trainer_mira import SentimentModelTrainerMIRA
from sentiment_model_trainer_swvm import SentimentModelTrainerSWVM
from sentiment_model_trainer_base import Corpus, CorpusFeaturesExtractor, SentimentModelConfiguration
from utils import Singleton


class SentimentModelTrainerFactory(Singleton):
    trainer_algorithms_dict = {
        'mira': SentimentModelTrainerMIRA,
        'SWVM': SentimentModelTrainerSWVM
    }

    def create_trainer(self, train_corpus: Corpus, features_extractor: CorpusFeaturesExtractor,
                       model_config: SentimentModelConfiguration, *args, **kwargs):

        if model_config.trainer_alg not in self.trainer_algorithms_dict:
            raise ValueError('`{}` is not a valid training algorithm type. Supported types are: {}'.format(
                model_config.trainer_alg,
                list(self.trainer_algorithms_dict.keys())
            ))

        trainer_alg_type_constructor = self.trainer_algorithms_dict[model_config.trainer_alg]
        return trainer_alg_type_constructor(train_corpus, features_extractor, model_config, *args, **kwargs)
