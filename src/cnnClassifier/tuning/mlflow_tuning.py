"""MLflow-aware HyperModel wrapper para Keras Tuner."""

import gc
import keras_tuner as kt
import tensorflow as tf
import mlflow
from loguru import logger


class MLflowHyperModel(kt.HyperModel):
    """
    Wrapper para kt.HyperModel que loga cada trial no MLflow.

    Sobrescreve o método fit() para criar uma child run no MLflow
    para cada trial, registrando hiperparâmetros e métricas.
    """

    def __init__(self, build_fn, experiment_name: str = None):
        """
        Args:
            build_fn: Função que constrói o modelo (recebe hp como argumento)
            experiment_name: Nome do experimento MLflow (opcional)
        """
        super().__init__()
        self.build_fn = build_fn
        self.experiment_name = experiment_name
        self.trial_counter = 0  # Contador para nomear trials sequencialmente

    def build(self, hp):
        """Delega construção do modelo para a função externa."""
        return self.build_fn(hp)

    def fit(self, hp, model, *args, **kwargs):
        """
        Sobrescreve fit() para logar cada trial no MLflow.

        Args:
            hp: HyperParameters do Keras Tuner
            model: Modelo compilado
            *args, **kwargs: Argumentos passados para model.fit()
        """
        # Incrementar contador e criar nome sequencial do trial
        self.trial_counter += 1
        trial_name = f"trial_{self.trial_counter}"

        try:
            # Criar child run para este trial
            with mlflow.start_run(nested=True, run_name=trial_name) as trial_run:
                # Logar hiperparâmetros do trial
                mlflow.log_params(hp.values)

                # Log trial_id para rastreamento
                if hasattr(hp, "trial_id"):
                    mlflow.log_param("trial_id", hp.trial_id)

                logger.info(f"📝 MLflow trial logged: {trial_run.info.run_id}")

                # Ativar autolog do TensorFlow (loga métricas, model, etc)
                mlflow.tensorflow.autolog(log_models=False, log_datasets=False)

                # Executar treinamento normal
                history = model.fit(*args, **kwargs)

                # Logar métricas finais do trial
                if hasattr(history, "history"):
                    for metric_name, values in history.history.items():
                        if values:
                            # Log última métrica de cada época
                            mlflow.log_metric(f"final_{metric_name}", values[-1])

                return history
        finally:
            # 🧹 Limpeza obrigatória de memória GPU + CPU no fim de cada trial
            try:
                tf.keras.backend.clear_session()
                # Reset de memória stats GPU (TF 2.11+)
                gpus = tf.config.list_physical_devices("GPU")
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.reset_memory_stats(gpu.name)
                        logger.debug(f"🧹 Memória GPU {gpu.name} resetada")
            except Exception as e:
                logger.debug(f"⚠️ Erro ao limpar GPU: {e}")
            finally:
                gc.collect()
                logger.debug(f"🧹 Limpeza completa após trial {trial_name}")
            logger.debug(f"🧹 Memória limpa após trial {trial_name}")
