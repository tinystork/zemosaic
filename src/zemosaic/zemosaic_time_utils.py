# -*- coding: utf-8 -*-
"""
A simple, reusable ETA calculator.
"""
import time
import math

class ETACalculator:
    """
    A simple helper class to calculate the Estimated Time of Arrival (ETA).
    """
    def __init__(self, total_items: int):
        """
        Initializes the ETA calculator.

        Args:
            total_items (int): The total number of items to be processed.
        """
        if total_items <= 0:
            raise ValueError("Total items must be a positive integer.")
        self.total_items = total_items
        self.processed_items = 0
        self.start_time = time.monotonic()

    def update(self, processed_items: int):
        """
        Updates the number of processed items.

        Args:
            processed_items (int): The new total of processed items.
        """
        self.processed_items = max(0, processed_items)

    def get_eta_seconds(self) -> float | None:
        """
        Calculates the estimated remaining time in seconds.

        Returns:
            Optional[float]: The ETA in seconds, or None if it cannot be determined.
        """
        if self.processed_items <= 0:
            return None

        elapsed_time = time.monotonic() - self.start_time
        time_per_item = elapsed_time / self.processed_items
        remaining_items = self.total_items - self.processed_items
        
        if remaining_items <= 0:
            return 0.0

        eta = remaining_items * time_per_item
        return eta if math.isfinite(eta) else None

    def get_progress(self) -> float:
        """
        Calculates the current progress as a fraction.

        Returns:
            float: The progress from 0.0 to 1.0.
        """
        if self.total_items == 0:
            return 1.0
        return min(1.0, self.processed_items / self.total_items)


def format_eta_hms(seconds: float, *, prefix: str = "") -> str:
    """
    Format ``seconds`` into an hh:mm:ss string with an optional prefix.

    Negative values automatically gain a ``+`` prefix if no prefix is provided.
    """
    try:
        total = float(seconds)
    except Exception:
        total = 0.0
    prefix_value = prefix
    if total < 0:
        prefix_value = prefix_value or "+"
        total = abs(total)
    eta_h, eta_rem = divmod(int(total + 0.5), 3600)
    eta_m, eta_s = divmod(eta_rem, 60)
    return f"{prefix_value}{eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
