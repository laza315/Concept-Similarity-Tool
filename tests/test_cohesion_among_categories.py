from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app

client = TestClient(app)


def test_compare_with_valid_inputs():
    response = client.get("/concepts_comparison/Book/Wallet?predictive_query=money&page=1&page_size=10")
    assert response.status_code == 200
    assert response.json() ==  {
                                # I'll most probably have to mock this
                               }

