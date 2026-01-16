"""Utility functions for the sample project."""

import hashlib
import re
from typing import Optional


def compute_hash(data: str) -> str:
    """Compute SHA256 hash of input string.
    
    Args:
        data: Input string to hash.
        
    Returns:
        Hexadecimal hash string.
    """
    return hashlib.sha256(data.encode()).hexdigest()


def validate_email(email: str) -> bool:
    """Validate email format.
    
    Args:
        email: Email address to validate.
        
    Returns:
        True if valid format.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def format_bytes(size: int) -> str:
    """Format byte count as human-readable string.
    
    Args:
        size: Size in bytes.
        
    Returns:
        Formatted string (e.g., "1.5 MB").
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


class Calculator:
    """Simple arithmetic calculator.
    
    Provides basic arithmetic operations with type hints.
    
    Example:
        >>> calc = Calculator()
        >>> calc.add(2, 3)
        5
    """
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b
    
    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        """Divide a by b.
        
        Args:
            a: Dividend.
            b: Divisor.
            
        Returns:
            Result of division.
            
        Raises:
            ValueError: If b is zero.
        """
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
