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
    if model_name in ["Adv-ASLTM", "CNNPred", 'CNNPred2D']:
        assert isinstance(predictions, np.ndarray)
        return predictions

    if model_name in ["THGNN", 'CNNPred3D']:
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


def convert_classification_preds(pred_paths: list[str], model_name: str) -> list[str]:
    converted_preds_paths = []

    for pred_file in pred_paths:
        converted_path = pred_file.replace("/results/", "/results_converted/")
        os.makedirs(os.path.dirname(converted_path), exist_ok=True)

        # Load original predictions
        with open(pred_file, 'rb') as f:
            results = pickle.load(f)

        results['preds'] = convert_predictions_to_rankings(model_name=model_name, predictions=results['preds'])

        # Save converted predictions
        with open(converted_path, 'wb') as f:
            pickle.dump(results, f)

        converted_preds_paths.append(converted_path)
        print(f'  ✓ Converted {pred_file} - {converted_path}')

    print(f'Completed {model_name} ({len(pred_paths)} files)\n')
    return converted_preds_paths


def convert_daily_to_cumulative_returns(preds_paths: list[str], sl: int, pl: int):
    x = pl
    if pl == 2 or pl == 6:
        x = x - pl

    converted_paths = []

    for pred_file in preds_paths:
        converted_path = pred_file.replace("/results/", "/results_converted/")
        os.makedirs(os.path.dirname(converted_path), exist_ok=True)

        # Load original predictions
        with open(pred_file, 'rb') as f:
            results = pickle.load(f)

        cumulative_returns = np.prod(1 + results['preds'][:, :x], axis=1) - 1
        results['preds'] = cumulative_returns[:, np.newaxis]
        cumulative_returns = np.prod(1 + results['labels'][:, :x], axis=1) - 1
        results['labels'] = cumulative_returns[:, np.newaxis]
        results['pred_date'] = [dt[x - 1] for dt in results['pred_date']]

        # Save converted results
        with open(converted_path, 'wb') as f:
            pickle.dump(results, f)

        converted_paths.append(converted_path)
        print(f'  ✓ Converted {pred_file} - {converted_path}')

    print(f'Completed ({len(preds_paths)} files)\n')
    return converted_paths, pl


