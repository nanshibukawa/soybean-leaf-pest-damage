import os

# 🔇 SILENCIA LOGS VERBOSOS DO TENSORFLOW 0=all, 1=info, 2=warning, 3=error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Silencia logs verbosos
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Remove warnings oneDNN
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import tensorflow as tf

from cnnClassifier.config.constants import DATA_SOURCE_DIR, DATA_TEST_DIR
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
    resize_with_pad: bool = True
    _cached_splits: tuple = field(default=None, init=False, repr=False)
    class_names: list = field(default_factory=list, init=False)

    def _create_padded_dataset(self, directory, class_names, target_class_names=None, shuffle=True, seed=None):
        directory = Path(directory)
        file_paths = []
        labels = []
        
        if target_class_names is None:
            target_class_names = class_names
            
        class_to_idx = {name: idx for idx, name in enumerate(target_class_names)}
        
        for name_in_dir, target_name in zip(class_names, target_class_names):
            class_dir = directory / name_in_dir
            if not class_dir.exists():
                matched_dir = None
                for d in directory.iterdir():
                    if d.is_dir() and (d.name == name_in_dir or d.name.replace('_', '-').startswith(name_in_dir.replace('_', '-'))):
                        matched_dir = d
                        break
                if matched_dir:
                    class_dir = matched_dir
                else:
                    logger.warning(f"⚠️ Pasta para classe {name_in_dir} não encontrada em {directory}")
                    continue
            
            idx = class_to_idx[target_name]
            for img_path in class_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    file_paths.append(str(img_path))
                    labels.append(idx)
                    
        if not file_paths:
            raise ValueError(f"Nenhuma imagem encontrada no diretório {directory} para as classes especificadas.")
            
        num_classes = len(target_class_names)
        one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
        
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, one_hot_labels))
        
        if shuffle:
            if seed is not None:
                dataset = dataset.shuffle(buffer_size=len(file_paths), seed=seed, reshuffle_each_iteration=True)
            else:
                dataset = dataset.shuffle(buffer_size=len(file_paths), reshuffle_each_iteration=True)
                
        target_height = self.image_config.altura
        target_width = self.image_config.largura
        
        def load_and_resize(file_path, label):
            img_bytes = tf.io.read_file(file_path)
            img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
            img.set_shape([None, None, 3])
            img = tf.cast(img, tf.float32)
            img = tf.image.resize_with_pad(img, target_height, target_width)
            return img, label
            
        dataset = dataset.map(load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.data_split_config.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def _load_all_splits(self):
        """
        Carrega dataset completo e divide em treino/validação/teste usando take/skip.
        Ou carrega diretamente as pastas 'train' e 'val' se existirem no diretório.
        Isso é feito uma única vez para eficiência.

        Returns:
            tuple: (train_dataset, validation_dataset, test_dataset)
        """
        if self._cached_splits is not None:
            logger.debug("♻️ Usando splits em cache")
            return self._cached_splits

        train_dir = Path(self.image_config.data_dir) / "train"
        val_dir = Path(self.image_config.data_dir) / "val"

        if train_dir.exists() and val_dir.exists():
            logger.info(f"📂 Detectado dataset pré-dividido em: {self.image_config.data_dir}")
            
            # Obter nomes das classes ordenados
            self.class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
            
            if self.resize_with_pad:
                logger.info("📐 Utilizando redimensionamento com Aspect-Ratio Padding (Letterbox)")
                with tf.device('/CPU:0'):
                    train_data = self._create_padded_dataset(
                        directory=train_dir,
                        class_names=self.class_names,
                        shuffle=True,
                        seed=self.data_split_config.random_seed
                    )
                    
                    validation_data = self._create_padded_dataset(
                        directory=val_dir,
                        class_names=self.class_names,
                        shuffle=False
                    )
                    
                    test_data = validation_data
                    
                    train_batches = tf.data.experimental.cardinality(train_data).numpy()
                    val_batches = tf.data.experimental.cardinality(validation_data).numpy()
            else:
                with tf.device('/CPU:0'):
                    # Carregar treino original (stretching)
                    train_data = tf.keras.utils.image_dataset_from_directory(
                        directory=train_dir,
                        label_mode="categorical",
                        shuffle=True,
                        seed=self.data_split_config.random_seed,
                        image_size=(self.image_config.altura, self.image_config.largura),
                        batch_size=self.data_split_config.batch_size,
                    )
                    self.class_names = train_data.class_names

                    # Carregar validação original
                    validation_data = tf.keras.utils.image_dataset_from_directory(
                        directory=val_dir,
                        label_mode="categorical",
                        shuffle=False,
                        class_names=self.class_names,
                        image_size=(self.image_config.altura, self.image_config.largura),
                        batch_size=self.data_split_config.batch_size,
                    )

                    test_data = validation_data

                    train_batches = tf.data.experimental.cardinality(train_data).numpy()
                    val_batches = tf.data.experimental.cardinality(validation_data).numpy()
            
            batch_size = self.data_split_config.batch_size
            train_images = train_batches * batch_size
            val_images = val_batches * batch_size

            self._cached_splits = (train_data, validation_data, test_data)

            logger.info(
                f"📊 Dataset Pré-Dividido Carregado: Train={train_images} imgs (aprox, {train_batches} batches) | "
                f"Val={val_images} imgs (aprox, {val_batches} batches)"
            )
        else:
            logger.info(f"📂 Carregando dataset único e realizando divisão dinâmica: {self.image_config.data_dir}")
            
            if self.resize_with_pad:
                logger.info("📐 Utilizando redimensionamento com Aspect-Ratio Padding (Letterbox) na divisão dinâmica")
                self.class_names = sorted([d.name for d in Path(self.image_config.data_dir).iterdir() if d.is_dir()])
                
                directory = Path(self.image_config.data_dir)
                file_paths = []
                labels = []
                
                class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
                for class_name in self.class_names:
                    class_dir = directory / class_name
                    idx = class_to_idx[class_name]
                    for img_path in class_dir.iterdir():
                        if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                            file_paths.append(str(img_path))
                            labels.append(idx)
                            
                num_classes = len(self.class_names)
                one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
                
                full_dataset = tf.data.Dataset.from_tensor_slices((file_paths, one_hot_labels))
                full_dataset = full_dataset.shuffle(buffer_size=len(file_paths), seed=self.data_split_config.random_seed, reshuffle_each_iteration=False)
                
                dataset_size = len(file_paths)
                train_size = int(dataset_size * self.data_split_config.train_ratio)
                val_size = int(dataset_size * self.data_split_config.val_ratio)
                
                train_paths_labels = full_dataset.take(train_size)
                remaining = full_dataset.skip(train_size)
                val_paths_labels = remaining.take(val_size)
                test_paths_labels = remaining.skip(val_size)
                
                target_height = self.image_config.altura
                target_width = self.image_config.largura
                batch_size = self.data_split_config.batch_size
                
                def load_and_resize(file_path, label):
                    img_bytes = tf.io.read_file(file_path)
                    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
                    img.set_shape([None, None, 3])
                    img = tf.cast(img, tf.float32)
                    img = tf.image.resize_with_pad(img, target_height, target_width)
                    return img, label
                    
                train_data = train_paths_labels.map(load_and_resize, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
                validation_data = val_paths_labels.map(load_and_resize, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
                test_data = test_paths_labels.map(load_and_resize, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
                
                train_batches = tf.data.experimental.cardinality(train_data).numpy()
                val_batches = tf.data.experimental.cardinality(validation_data).numpy()
                test_batches = tf.data.experimental.cardinality(test_data).numpy()
                
                train_images = train_batches * batch_size
                val_images = val_batches * batch_size
                test_images = test_batches * batch_size
            else:
                # Carregar dataset completo original (stretching)
                with tf.device('/CPU:0'):
                    full_dataset = tf.keras.utils.image_dataset_from_directory(
                        directory=self.image_config.data_dir,
                        label_mode="categorical",
                        shuffle=True,
                        seed=self.data_split_config.random_seed,
                        image_size=(self.image_config.altura, self.image_config.largura),
                        batch_size=self.data_split_config.batch_size,
                    )
                    self.class_names = full_dataset.class_names

                    # Calculo do tamanho (em batches)
                    dataset_size_batches = tf.data.experimental.cardinality(full_dataset).numpy()
                    train_batches = int(dataset_size_batches * self.data_split_config.train_ratio)
                    val_batches = int(dataset_size_batches * self.data_split_config.val_ratio)
                    test_batches = int(dataset_size_batches * self.data_split_config.test_ratio)

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
                f"📊 Divisão Dinâmica: Train={train_images} imgs ({self.data_split_config.train_ratio*100:.0f}%) | "
                f"Val={val_images} ({self.data_split_config.val_ratio*100:.0f}%) | "
                f"Test={test_images} ({self.data_split_config.test_ratio*100:.0f}%)"
            )
            
        return self._cached_splits

    def load_train_data(self):
        """Carrega dados de treino (compatível com Stage 5)"""
        with tf.device('/CPU:0'):
            train_data, _, _ = self._load_all_splits()

            # O dataset já vem batchado pelo image_dataset_from_directory
            # CutMix operates on (images, labels) batches
            seed = self.data_split_config.random_seed
            cutmix = tf.keras.layers.CutMix(seed=seed)
            logger.info(f"CutMix habilitado com seed: {seed}")

            def apply_cutmix(images, labels):
                outputs = cutmix({"images": images, "labels": labels})
                return outputs["images"], outputs["labels"]

            train_data = train_data.map(apply_cutmix, num_parallel_calls=tf.data.AUTOTUNE)
            train_data = train_data.prefetch(tf.data.AUTOTUNE)
        return train_data

    def load_validation_data(self):
        """Carrega dados de validação (compatível com Stage 5)"""
        _, validation_data, _ = self._load_all_splits()
        return validation_data

    # def load_test_data(self):
    #     """Carrega dados de teste para avaliação final"""
    #     _, _, test_data = self._load_all_splits()
    #     return test_data

    def load_test_data(self):
        """Carrega dados de teste para avaliação final"""
        test_dir = Path("artifacts/data/final/INSECT12C-test")
        if test_dir.exists():
            logger.info(f"🧪 Carregando dataset de teste externo de: {test_dir}")
            
            # Garante que os splits principais foram gerados para carregar a ordem exata das classes
            if not self.class_names:
                self._load_all_splits()
                
            # Mapeia dinamicamente os nomes das pastas correspondentes no diretório de teste
            test_subdirs = [d.name for d in test_dir.iterdir() if d.is_dir()]
            mapped_test_class_names = []
            for class_name in self.class_names:
                matched = None
                for subdir in test_subdirs:
                    if subdir.startswith(class_name) or subdir.replace('_', '-').startswith(class_name.replace('_', '-')):
                        matched = subdir
                        break
                if matched:
                    mapped_test_class_names.append(matched)
                else:
                    logger.warning(f"⚠️ Não foi possível mapear a classe {class_name} para as pastas de teste. Usando fallback.")
                    mapped_test_class_names.append(class_name)

            if self.resize_with_pad:
                logger.info("📐 Utilizando redimensionamento com Aspect-Ratio Padding (Letterbox) no teste externo")
                with tf.device('/CPU:0'):
                    return self._create_padded_dataset(
                        directory=test_dir,
                        class_names=mapped_test_class_names,
                        target_class_names=self.class_names,
                        shuffle=False
                    )
            else:
                with tf.device('/CPU:0'):
                    return tf.keras.utils.image_dataset_from_directory(
                        directory=test_dir,
                        label_mode="categorical",
                        class_names=mapped_test_class_names,  # Garante mesmo mapeamento de classes
                        shuffle=False,
                        image_size=(self.image_config.altura, self.image_config.largura),
                        batch_size=self.data_split_config.batch_size,
                    )
        else:
            logger.warning(f"⚠️ Diretório de teste {test_dir} não encontrado. Usando split original.")
            _, _, test_data = self._load_all_splits()
            return test_data


    def load_all_splits(self):
        """Retorna os 3 splits de uma vez (mais eficiente se precisa de todos)"""
        return self._load_all_splits()

    def get_class_distribution(self):
        """Calcula a distribuição por classe em train/val/test."""
        train_data, validation_data, test_data = self._load_all_splits()

        train_dir = Path(self.image_config.data_dir) / "train"
        val_dir = Path(self.image_config.data_dir) / "val"
        if train_dir.exists() and val_dir.exists():
            class_dir = train_dir
        else:
            class_dir = Path(self.image_config.data_dir)

        with tf.device('/CPU:0'):
            class_names = tf.keras.utils.image_dataset_from_directory(
                directory=class_dir,
                shuffle=False,
                image_size=(self.image_config.altura, self.image_config.largura),
                batch_size=1,
            ).class_names
        num_classes = len(class_names)

        def count_labels(ds):
            counts = np.zeros(num_classes, dtype=int)
            for _, y in ds:
                y = y.numpy()
                y_indices = np.argmax(
                    y, axis=-1
                )  # <--- Reverte o One-Hot para o índice escalar
                for i in range(num_classes):
                    counts[i] += (y_indices == i).sum()
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

    # def get_class_weights(self):
    #     """Calcula pesos de classe baseados na distribuição do conjunto de treino."""
    #     dist = self.get_class_distribution()
    #     train_stats = dist["train"]
    #     total = train_stats["total"]
    #     counts = train_stats["counts"]
    #     class_names = dist["class_names"]
    #     num_classes = len(class_names)
    # 
    #     class_weights = {}
    #     for idx, class_name in enumerate(class_names):
    #         count = counts[class_name]
    #         if count > 0:
    #             # Fórmula padrão: total / (num_classes * count)
    #             weight = total / (num_classes * count)
    #         else:
    #             weight = 1.0
    #         class_weights[idx] = float(weight)
    # 
    #     logger.info(f"⚖️ Pesos de classe dinâmicos calculados: {class_weights}")
    #     return class_weights




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
