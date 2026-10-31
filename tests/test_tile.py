import numpy as np

from invisimark import tile

KEY = b"unit-test-key-0123456789"


def test_tiles_are_deterministic_and_keyed():
    a = tile.id_tile(KEY, "alice")
    assert np.array_equal(a, tile.id_tile(KEY, "alice"))
    assert not np.array_equal(a, tile.id_tile(b"another-key-another-key", "alice"))
    assert not np.array_equal(a, tile.id_tile(KEY, "bob"))


def test_tile_is_zero_mean_unit_variance_and_band_limited():
    t = tile.id_tile(KEY, "alice")
    assert abs(float(t.mean())) < 1e-4
    assert abs(float(t.std()) - 1.0) < 1e-3
    power = np.abs(np.fft.rfft2(t)) ** 2
    fy = np.fft.fftfreq(tile.TILE) * tile.TILE
    fx = np.fft.rfftfreq(tile.TILE) * tile.TILE
    radius = np.hypot(fy[:, None], fx[None, :])
    outside = power[(radius < 0.9) | (radius > tile.DEFAULT.band_hi + 1.1)].sum()
    assert outside / power.sum() < 1e-6


def test_distinct_recipients_are_nearly_orthogonal():
    tiles = np.stack([tile.id_tile(KEY, f"user{i}") for i in range(200)]).reshape(200, -1)
    gram = tiles @ tiles.T / tiles.shape[1]
    off_diagonal = gram[~np.eye(200, dtype=bool)]
    assert np.abs(off_diagonal).max() < 0.15
    assert abs(float(tile.sync_tile(KEY).ravel() @ tiles[0]) / tiles.shape[1]) < 0.15


def test_plane_tiles_seamlessly():
    plane = tile.tile_to_plane(tile.id_tile(KEY, "alice"), 300, 700)
    assert plane.shape == (300, 700)
    assert np.array_equal(plane[:, :128], plane[:, 128:256])


def test_slugs_never_collide():
    from invisimark import keys

    stems = keys.unique_slugs(["Bob!", "bob?", "bob", "Alice", "alice"])
    assert len(set(stems.values())) == 5
    assert stems["Alice"] == "alice" and stems["Bob!"] == "bob"
