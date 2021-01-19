import argparse
from pathlib import Path
from random import seed

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from trains import Task

import conf
from models import io as models_io
from src import data_io

# <editor-fold desc="Argument Parsing">
parser = argparse.ArgumentParser(description='K Fold Evaluation top N accuracy, N ∈ {1, 4, 10, 50}',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # <- Includes defaults in docs

parser.add_argument('model',
                    type=str,
                    help='Model name to train')

parser.add_argument('--enqueue',
                    type=str,
                    default='',
                    help='Queue name for remote execution. Defaults to an empty string and executes locally')

parser.add_argument('--random_seed',
                    type=int,
                    default=1729,
                    help='Random seed')

ARGS = parser.parse_args()


# </editor-fold>

def main():
    # Init task
    task = Task.init(project_name=conf.PROJECT_NAME,
                     task_name='Test Set Evaluation',
                     task_type=Task.TaskTypes.testing)

    root_data_dir = conf.DATA_DIR

    # Execute locally or remotely
    if ARGS.enqueue:
        task.execute_remotely(queue_name=ARGS.enqueue)
        root_data_dir = Path("/opt/data")

    # Prerequisites
    np.random.seed(ARGS.random_seed)
    seed(ARGS.random_seed)
    logger = task.get_logger()
    true_ranks = dict()
    preprocess_pipeline, to_dataset, get_model = models_io.get_fitting_method(ARGS.model)

    # Load data, limit to at least 4 cities (the minimum in the real test set)
    train_data, validation_data, test_data = data_io.load_split_train_validation_test(data_dir=root_data_dir,
                                                                                      trip_length_threshold=4)

    # Prepare X and y
    train_x, train_y = data_io.separate_features_from_label(train_data)
    validation_x, validation_y = data_io.separate_features_from_label(validation_data)
    test_x, test_y = data_io.separate_features_from_label(test_data)

    # Label encoder
    label_encoder = LabelEncoder().fit(train_y['city_id'].tolist() +
                                       test_y['city_id'].tolist() +
                                       validation_y['city_id'].tolist())

    # Encode cities
    train_y.loc[:, 'city_id'] = label_encoder.transform(train_y['city_id'])
    validation_y.loc[:, 'city_id'] = label_encoder.transform(validation_y['city_id'])
    test_y.loc[:, 'city_id'] = label_encoder.transform(test_y['city_id'])

    # Prepare preprocess pipeline
    print('Fitting preprocess pipeline...')
    pp_pipeline = preprocess_pipeline(features=train_x, labels=train_y)

    # Transform features
    print('Transforming features...')
    train_x_processed = pp_pipeline.transform(train_x)
    validation_x_processed = pp_pipeline.transform(validation_x)
    test_x_processed = pp_pipeline.transform(test_x)

    # Prepare dataset objects
    train_dataset = to_dataset(train_x_processed, train_y, batch_size=32, pre_shuffle=True)
    validation_dataset = to_dataset(validation_x_processed, validation_y, batch_size=1000, pre_shuffle=False)
    test_dataset = to_dataset(test_x_processed, test_y, batch_size=1000, pre_shuffle=False)

    # Create model
    most_probable_cities_keys = [k for k in train_x_processed.columns if k.endswith('most_probable_city_id')]
    n_cities = len(pp_pipeline['encode'].transformers_[0][1].categories_[0]) + 1  # Plus 1 for unknown
    n_target_cities = max(len(pp_pipeline['encode'].transformers_[0][1].categories_[n]) for n in
                          range(5, 5 + len(most_probable_cities_keys)))
    n_devices = len(pp_pipeline['encode'].transformers_[0][1].categories_[1])
    n_affiliates = len(pp_pipeline['encode'].transformers_[0][1].categories_[2]) + 1
    n_booker_countries = len(pp_pipeline['encode'].transformers_[0][1].categories_[3])
    n_hotel_countries = len(pp_pipeline['encode'].transformers_[0][1].categories_[4]) + 1
    n_labels = len(label_encoder.classes_)
    model = get_model(n_cities=n_cities,
                      n_target_cities=n_target_cities,
                      n_devices=n_devices,
                      n_affiliates=n_affiliates,
                      n_booker_countries=n_booker_countries,
                      n_hotel_countries=n_hotel_countries,
                      n_labels=n_labels,
                      most_probable_cities_keys=most_probable_cities_keys)
    model.fit(train_dataset,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=3,
                                                          monitor='val_sparse_top_k_categorical_accuracy'),
                         tf.keras.callbacks.TensorBoard(write_graph=False)],
              validation_data=validation_dataset,
              epochs=200)

    # Evaluate Top N accuracy, N ∈ {1, 4, 10, 50}
    probabilities = model.predict(test_dataset)
    y_true = tf.one_hot(test_y['city_id'].values, depth=len(label_encoder.classes_))
    for n in [1, 4, 10, 50]:
        # Evaluate
        top_n_accuracy = tf.keras.metrics.top_k_categorical_accuracy(y_true, probabilities, k=n).numpy().mean()

        # Report accuracy
        logger.report_scalar(title='Top K Accuracy',
                             series=f'{ARGS.model}',
                             iteration=n,
                             value=top_n_accuracy)

    # Gather rank of true label for later analysis
    true_ranks['true_label_rank'] = np.where(np.argsort(-probabilities, axis=1) ==
                                             test_y.city_id.values.reshape(-1, 1))[1]
    true_ranks['utrip_id'] = test_y['utrip_id'].tolist()

    # Report metrics
    true_ranks = pd.DataFrame(true_ranks)
    task.upload_artifact(name='true_label_ranks',
                         artifact_object=true_ranks)


if __name__ == '__main__':
    main()
