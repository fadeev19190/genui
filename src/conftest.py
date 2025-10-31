import pytest
from django.db import connections

@pytest.fixture(scope="session", autouse=True)
def close_db_connections_on_exit():
    yield
    connections.close_all()