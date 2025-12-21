import numpy as np

class LabelUtils:
    """Centralized logic for handling partial labels across domains."""
    
    @staticmethod
    def get_unlabeled_mask(y: np.ndarray) -> np.ndarray:
        """
        Returns boolean mask: True if value is missing (-1, NaN, None).
        Handles int, float, and object arrays robustly.
        """
        y = np.asarray(y).ravel()
        mask = np.zeros(y.shape[0], dtype=bool)

        # 1. Sentinel check (-1)
        # We wrap in try/except because strict string arrays might fail comparison
        try:
            mask |= (y == -1)
        except Exception:
            pass

        # 2. NaN / None check
        if np.issubdtype(y.dtype, np.number):
            mask |= np.isnan(y)
        else:
            # Object/String arrays: check None and NaN-like objects
            for i, v in enumerate(y):
                if v is None:
                    mask[i] = True
                else:
                    try:
                        if v != v:  # Standard NaN check
                            mask[i] = True
                    except Exception:
                        pass
        return mask

    @staticmethod
    def get_valid_classes(y: np.ndarray) -> np.ndarray:
        """Returns sorted unique labels, excluding missing values."""
        y = np.asarray(y).ravel()
        mask = LabelUtils.get_unlabeled_mask(y)
        return np.unique(y[~mask])

    @staticmethod
    def validate_shared_labels(y_a: np.ndarray, y_b: np.ndarray, strict=True):
        """
        Ensures both domains share the same semantic label space.
        Returns the sorted list of valid shared classes.
        """
        labels_a = LabelUtils.get_valid_classes(y_a)
        labels_b = LabelUtils.get_valid_classes(y_b)

        set_a, set_b = set(labels_a), set(labels_b)
        
        if strict and set_a != set_b:
            only_a = sorted(set_a - set_b)
            only_b = sorted(set_b - set_a)
            raise ValueError(
                "Domain label mismatch (shared semantic space violated).\n"
                f"  Labels only in A: {only_a}\n"
                f"  Labels only in B: {only_b}"
            )
        
        # If not strict, we return the union of known classes
        return np.array(sorted(set_a | set_b))