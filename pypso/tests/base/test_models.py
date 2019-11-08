import pytest

from pypso.base import BasePSO

def test_base():
    """Tests BasePSO.
    """
    with pytest.raises(TypeError):
        BasePSO()