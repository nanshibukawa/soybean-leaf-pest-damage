import os

# 🔇 SILENCIA LOGS VERBOSOS DO TENSORFLOW 0=all, 1=info, 2=warning, 3=error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Silencia logs verbosos
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Remove warnings oneDNN
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import tensorflow as tf

from cnnClassifier.config.constants import DATA_SOURCE_DIR
from cnnClassifier.entity.config_entity import (
    DataSplitterConfig,
    ImageConfig,
    DataSubsetType,
)
from cnnClassifier.utils.logger import configure_logger

logger = configure_logger(__name__)


@dataclass
class DataSplitter:
    """Divide dados de imagem em conjuntos de treino e validação.

    Atributos:
        data_split_config (DataSplitterConfig): Configuração para proporções de divisão
            de dados, tamanho do batch e random seed para divisões reproduzíveis
        image_config (ImageConfig): Configuração para dimensões de imagem (altura/largura)
            e o caminho do diretório de dados
        subset (DataSubsetType): Definições de tipo para subconjuntos de dados (TRAIN/VALIDATION)
            usado para especificar qual porção dos dados carregar
    """

    data_split_config: DataSplitterConfig
    image_config: ImageConfig
    subset: DataSubsetType
    _cached_splits: tuple = field(default=None, init=False, repr=False)

    def _load_all_splits(self):
        """
        Carrega dataset completo e divide em treino/validação/teste usando take/skip.
        Isso é feito uma única vez para eficiência.

        Returns:
            tuple: (train_dataset, validation_dataset, test_dataset)
        """
        if self._cached_splits is not None:
            logger.debug("♻️ Usando splits em cache")
            return self._cached_splits
        # Carregar dataset completo
        full_dataset = tf.keras.utils.image_dataset_from_directory(
            directory=self.image_config.data_dir,
            shuffle=True,
            seed=self.data_split_config.random_seed,
            image_size=(self.image_config.altura, self.image_config.largura),
            batch_size=self.data_split_config.batch_size,
        )

        # Calculo do tamanho (em batches)
        dataset_size_batches = tf.data.experimental.cardinality(full_dataset).numpy()
        train_batches = int(dataset_size_batches * self.data_split_config.train_ratio)
        val_batches = int(dataset_size_batches * self.data_split_config.val_ratio)
        test_batches = int(dataset_size_batches * self.data_split_config.test_ratio)

        # Calcular número de imagens (aproximado)
        batch_size = self.data_split_config.batch_size
        train_images = train_batches * batch_size
        val_images = val_batches * batch_size
        test_images = test_batches * batch_size

        # Dividir: treino | validação | teste
        train_data = full_dataset.take(train_batches)
        remaining = full_dataset.skip(train_batches)
        validation_data = remaining.take(val_batches)
        test_data = remaining.skip(val_batches)

        self._cached_splits = (train_data, validation_data, test_data)

        logger.info(
            f"📊 Divisão: Train={train_images} imgs ({self.data_split_config.train_ratio*100:.0f}%) | "
            f"Val={val_images} ({self.data_split_config.val_ratio*100:.0f}%) | "
            f"Test={test_images} ({self.data_split_config.test_ratio*100:.0f}%)"
        )
        return self._cached_splits

    def load_train_data(self):
        """Carrega dados de treino (compatível com Stage 5)"""
        train_data, _, _ = self._load_all_splits()
        return train_data

    def load_validation_data(self):
        """Carrega dados de validação (compatível com Stage 5)"""
        _, validation_data, _ = self._load_all_splits()
        return validation_data

    def load_test_data(self):
        """Carrega dados de teste para avaliação final"""
        _, _, test_data = self._load_all_splits()
        return test_data

    def load_all_splits(self):
        """Retorna os 3 splits de uma vez (mais eficiente se precisa de todos)"""
        return self._load_all_splits()

    def get_class_distribution(self):
        """Calcula a distribuição por classe em train/val/test."""
        train_data, validation_data, test_data = self._load_all_splits()

        class_names = tf.keras.utils.image_dataset_from_directory(
            directory=self.image_config.data_dir,
            shuffle=False,
            image_size=(self.image_config.altura, self.image_config.largura),
            batch_size=1,
        ).class_names
        num_classes = len(class_names)

        def count_labels(ds):
            counts = np.zeros(num_classes, dtype=int)
            for _, y in ds:
                y = y.numpy()
                for i in range(num_classes):
                    counts[i] += (y == i).sum()
            return counts

        def summarize(ds):
            counts = count_labels(ds)
            total = int(counts.sum())
            return {
                "total": total,
                "counts": {class_names[i]: int(counts[i]) for i in range(num_classes)},
                "percentages": {
                    class_names[i]: (100 * counts[i] / total) if total else 0.0
                    for i in range(num_classes)
                },
            }

        return {
            "class_names": class_names,
            "train": summarize(train_data),
            "validation": summarize(validation_data),
            "test": summarize(test_data),
        }

    # def load_train_data(self):
    #     """
    #     Carrega dados de treino do diretório configurado.

    #     Cria um dataset TensorFlow para treino carregando imagens do diretório especificado
    #     e aplicando a proporção de divisão de treino configurada. O dataset é automaticamente
    #     embaralhado e agrupado em lotes de acordo com a configuração.

    #     Returns:
    #         tf.data.Dataset: Dataset de treino contendo imagens agrupadas em batchs e
    #             pré-processadas com seus rótulos correspondentes. Cada batch contém imagens
    #             redimensionadas para as dimensões configuradas (altura x largura) e o
    #             dataset usa o subconjunto de treino baseado na proporção de divisão.
    #     """
    #     treino = tf.keras.utils.image_dataset_from_directory(
    #         directory=self.image_config.data_dir,
    #         validation_split=1 - self.data_split_config.train_ratio,
    #         shuffle=True,
    #         subset=self.subset.TRAIN.value,
    #         seed=self.data_split_config.random_seed,
    #         image_size=(self.image_config.altura, self.image_config.largura),
    #         batch_size=self.data_split_config.batch_size,
    #     )
    #     return treino

    # def load_validation_data(self):
    #     """Carrega dados de validação do diretório configurado.

    #     Cria um dataset TensorFlow para validação carregando imagens do diretório
    #     especificado e aplicando a proporção de divisão de validação configurada.
    #     O dataset é automaticamente embaralhado e agrupado em batchs de acordo com a
    #     configuração, usando a mesma semente dos dados de treino para divisões consistentes.

    #     Returns:
    #         tf.data.Dataset: Dataset de validação contendo imagens agrupadas em batchs e
    #             pré-processadas com seus rótulos correspondentes. Cada batch contém imagens
    #             redimensionadas para as dimensões configuradas (altura x largura) e o
    #             dataset usa o subconjunto de validação baseado na proporção de divisão.
    #     """
    #     validation = tf.keras.utils.image_dataset_from_directory(
    #         directory=self.image_config.data_dir,
    #         validation_split=self.data_split_config.val_ratio,
    #         shuffle=True,
    #         subset=self.subset.VALIDATION.value,
    #         seed=self.data_split_config.random_seed,
    #         image_size=(self.image_config.altura, self.image_config.largura),
    #         batch_size=self.data_split_config.batch_size,
    #     )
    #     return validation


if __name__ == "__main__":
    logger.info("Iniciando o DataSplitter...")

    data_split_config = DataSplitterConfig(
        batch_size=32,
        random_seed=42,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )

    #   size: [224, 224, 3]
    image_config = ImageConfig(
        altura=224, largura=224, canais=3, data_dir=Path(DATA_SOURCE_DIR)
    )

    data_splitter = DataSplitter(
        data_split_config=data_split_config,
        image_config=image_config,
        subset=DataSubsetType,
    )

    train_data = data_splitter.load_train_data()
    validation_data = data_splitter.load_validation_data()
    test_data = data_splitter.load_test_data()

    # Contar batches reais
    train_batches = tf.data.experimental.cardinality(train_data).numpy()
    val_batches = tf.data.experimental.cardinality(validation_data).numpy()
    test_batches = tf.data.experimental.cardinality(test_data).numpy()

    distribution = data_splitter.get_class_distribution()

    def show_split(name, stats):
        total = stats["total"]
        print(f"\n{name} (total={total})")
        for class_name in distribution["class_names"]:
            count = stats["counts"][class_name]
            pct = stats["percentages"][class_name]
            print(f"  {class_name}: {count} ({pct:.2f}%)")

    show_split("Train", distribution["train"])
    show_split("Validation", distribution["validation"])
    show_split("Test", distribution["test"])

    logger.info(
        f"✅ Splits carregados: "
        f"Train={train_batches} batches, "
        f"Val={val_batches} batches, "
        f"Test={test_batches} batches"
    )

    logger.info("Dados de treino e validação carregados com sucesso.")
