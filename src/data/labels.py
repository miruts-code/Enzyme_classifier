"""Label utilities: EC class integer -> short name."""
from config import EC_CLASS_NAMES


def label_to_name(y):
    """Integer EC class -> short name. Example: 3 -> 'Hydrolases'."""
    return EC_CLASS_NAMES.get(int(y), f"EC {y}")


def labels_to_names(ys):
    """List of integers -> list of short names."""
    return [label_to_name(y) for y in ys]
