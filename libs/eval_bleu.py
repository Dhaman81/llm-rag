from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def get_bleu_score(reference, prediction):
    reference_tokens = [reference.lower().split()]
    prediction_tokens = prediction.lower().split()

    smoothie = SmoothingFunction().method4

    scores = {
        'BLEU-1': sentence_bleu(reference_tokens, prediction_tokens, weights=(1, 0, 0, 0),
                                smoothing_function=smoothie),
        'BLEU-2': sentence_bleu(reference_tokens, prediction_tokens, weights=(0.5, 0.5, 0, 0),
                                smoothing_function=smoothie),
        'BLEU-3': sentence_bleu(reference_tokens, prediction_tokens, weights=(0.33, 0.33, 0.33, 0),
                                smoothing_function=smoothie),
        'BLEU-4': sentence_bleu(reference_tokens, prediction_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                                smoothing_function=smoothie),
    }

    return scores