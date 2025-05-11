def evaluate(query, expected, actual):
    expected = expected.lower()
    actual = actual.lower()
    tp = int(expected in actual)
    precision = tp
    recall = tp
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1
