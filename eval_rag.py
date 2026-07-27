#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from eval.evaluator import RAGEvaluator

RAGEvaluator().run()
