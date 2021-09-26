import pandas as pd
import pytest

def test_len_srs():
    assert len(pd.Series())==0