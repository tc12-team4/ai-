# database/__init__.py
"""
Package database pour Doxa Support AI
"""

from .results_db import ResultsDB, create_database

__all__ = ['ResultsDB', 'create_database']