"""Configuration pytest pour les tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ajouter le répertoire racine du projet au path pour les imports.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mock streamlit avant tous les imports.
mock_streamlit = MagicMock()
mock_streamlit.cache_resource = lambda *args, **kwargs: lambda func: func
mock_streamlit.cache_data = lambda *args, **kwargs: lambda func: func
sys.modules['streamlit'] = mock_streamlit

# Mock deep_translator avant tous les imports.
mock_deep_translator = MagicMock()
sys.modules['deep_translator'] = mock_deep_translator
sys.modules['deep_translator.google'] = MagicMock()
sys.modules['deep_translator.google.GoogleTranslator'] = MagicMock()

# Mock tensorflow pour éviter les imports lourds dans les tests.
mock_tensorflow = MagicMock()
mock_tensorflow.keras = MagicMock()
mock_tensorflow.keras.models = MagicMock()
mock_tensorflow.keras.models.load_model = MagicMock()
mock_tensorflow.linalg = MagicMock()
mock_tensorflow.linalg.l2_normalize = MagicMock()
mock_tensorflow.abs = MagicMock()
sys.modules['tensorflow'] = mock_tensorflow

