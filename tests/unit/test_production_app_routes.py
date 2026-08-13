from production_app import app


def test_governance_routes_are_mounted():
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/reviews" in paths
    assert "/reviews/claim" in paths
    assert "/feedback" in paths
