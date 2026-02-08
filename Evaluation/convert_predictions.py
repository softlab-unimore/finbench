import os
import pickle
import numpy as np
from scipy.special import softmax


def convert_predictions_to_rankings(
        model_name: str,
        predictions: list[np.ndarray] | np.ndarray
) -> list[np.ndarray] | np.ndarray:
    """ Convert classification predictions to portfolio ranking scores """

    # Already return ranking scores
    if model_name in ["Adv-ASLTM", "CNNPred"]:
        assert isinstance(predictions, np.ndarray)
        return predictions

    if model_name == "THGNN":
        assert isinstance(predictions, list)
        assert all([isinstance(arr, np.ndarray) and arr.ndim == 2 for arr in predictions])

        return predictions

    # Convert logits to buy probabilities
    if model_name in ["DGDNN", "MAN-SF"]:
        assert isinstance(predictions, list)
        assert all([isinstance(arr, np.ndarray) and arr.ndim == 2 for arr in predictions])

        return [softmax(arr, axis=1)[:, [1]] for arr in predictions]

    # Long/short strategy: positive scores for long, negative for short
    if model_name == "HGTAN":
        assert isinstance(predictions, list)
        assert all([isinstance(arr, np.ndarray) and arr.ndim == 2 for arr in predictions])

        results = []

        for logits in predictions:
            probs = softmax(logits, axis=1)
            predicted_class: np.ndarray = np.argmax(probs, axis=1)

            scores = np.full(len(logits), np.nan)
            scores[predicted_class == 1] = probs[predicted_class == 1, 1]  # Long
            scores[predicted_class == 0] = -probs[predicted_class == 0, 0]  # Short

            results.append(scores[:, np.newaxis])

        return results

    raise ValueError(f"Unknown model: {model_name}")


def convert_all_classification_predictions(source_dir: str, target_dir: str):
    """ Convert all prediction files to portfolio scores """

    # Create target directory
    os.makedirs(target_dir, exist_ok=True)

    # Process each model directory
    for model_name in os.listdir(source_dir):
        model_path = os.path.join(source_dir, model_name)

        # Skip if not a directory
        if not os.path.isdir(model_path):
            continue

        print(f'Processing {model_name}...')

        # Create corresponding target model directory
        target_model_path = os.path.join(target_dir, model_name)
        os.makedirs(target_model_path, exist_ok=True)

        # Find all result pickle files
        pred_files = [f for f in os.listdir(model_path) if f.startswith('results_') and f.endswith('.pkl')]

        # Convert each prediction file
        for pred_file in pred_files:
            source_file = os.path.join(model_path, pred_file)
            target_file = os.path.join(target_model_path, pred_file)

            # Load original results
            with open(source_file, 'rb') as f:
                results = pickle.load(f)

            # Convert predictions to portfolio scores
            results['preds'] = convert_predictions_to_rankings(model_name=model_name, predictions=results['preds'])

            # Save converted results
            with open(target_file, 'wb') as f:
                pickle.dump(results, f)

            print(f'  ✓ Converted {pred_file}')

        print(f'Completed {model_name} ({len(pred_files)} files)\n')


def convert_daily_to_cumulative_returns(model_path: str, target_model_path: str):
    """ Convert one D-Va daily returns to cumulative returns """
    if not os.path.isdir(model_path):
        raise ValueError(f"Model directory not found: {model_path}")

    print(f'Processing {model_path}...')

    # Create target directory
    os.makedirs(target_model_path, exist_ok=True)

    # Find all result pickle files
    pred_files = [f for f in os.listdir(model_path) if f.startswith('results_') and f.endswith('.pkl')]

    # Convert each prediction file
    for pred_file in pred_files:
        source_file = os.path.join(model_path, pred_file)
        target_file = os.path.join(target_model_path, pred_file)

        # Load original results
        with open(source_file, 'rb') as f:
            results = pickle.load(f)

        # Convert daily returns to cumulative returns
        cumulative_returns = np.prod(1 + results['preds'], axis=1) - 1
        results['preds'] = cumulative_returns[:, np.newaxis]
        cumulative_returns = np.prod(1 + results['labels'], axis=1) - 1
        results['labels'] = cumulative_returns[:, np.newaxis]
        results['pred_date'] = [dt[-1] for dt in results['pred_date']]

        # Save converted results
        with open(target_file, 'wb') as f:
            pickle.dump(results, f)

        print(f'  ✓ Converted {pred_file}')

    print(f'Completed ({len(pred_files)} files)\n')


if __name__ == '__main__':
    convert_all_classification_predictions(
        source_dir='./data/preds/new_res_finbech/Classification',
        target_dir='./data/preds/new_res_finbech/ClassificationConverted'
    )

    convert_daily_to_cumulative_returns(
        model_path='./data/preds/new_res_finbech/Regression/D-Va',
        target_model_path='./data/preds/new_res_finbech/Regression/D-Va_new',
    )
