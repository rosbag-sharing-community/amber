from amber_mcap.unit.time import Time, TimeUnit
from pytest import approx


def test_time_unit() -> None:
    sec = Time(0.1, TimeUnit.SECOND)
    assert sec.get(TimeUnit.MILLISECOND) == approx(100.0)
    assert sec.get(TimeUnit.MICROSECOND) == approx(100000.0)
    assert sec.get(TimeUnit.NANOSECOND) == approx(100000000.0)
    millisec = Time(0.1, TimeUnit.MILLISECOND)
    assert millisec.get(TimeUnit.SECOND) == approx(0.0001)
    assert millisec.get(TimeUnit.MILLISECOND) == approx(0.1)
    assert millisec.get(TimeUnit.MICROSECOND) == approx(100)
    assert millisec.get(TimeUnit.NANOSECOND) == approx(100000)
