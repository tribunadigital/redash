import datetime
import decimal
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[2] / "redash" / "query_runner" / "python_compat.py"
SPEC = importlib.util.spec_from_file_location("python_compat_standalone", MODULE_PATH)
compat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compat)


class Bomb:
    def __str__(self):
        raise AssertionError("__str__ must not be called")

    def __repr__(self):
        raise AssertionError("__repr__ must not be called")

    def __format__(self, spec):
        raise AssertionError("__format__ must not be called")


class StringSubclass(str):
    pass


class MappingBomb:
    def __getitem__(self, key):
        raise AssertionError("mapping hook must not be called")


class CollisionKey:
    events = []

    def __hash__(self):
        return hash("name")

    def __eq__(self, other):
        self.events.append(other)
        return False


class EqualityBombMeta(type):
    events = []

    def __eq__(cls, other):
        cls.events.append(other)
        return other is list


class EqualityBomb(metaclass=EqualityBombMeta):
    def __repr__(self):
        raise AssertionError("__repr__ must not be called")

    def __format__(self, spec):
        raise AssertionError("__format__ must not be called")


class LocalTimezoneProbe(datetime.tzinfo):
    events = []

    def utcoffset(self, value):
        self.events.append("utcoffset")
        return datetime.timedelta()

    def dst(self, value):
        self.events.append("dst")
        return datetime.timedelta()

    def tzname(self, value):
        self.events.append("tzname")
        return "probe"

    def __repr__(self):
        self.events.append("repr")
        raise AssertionError("timezone repr must not be called")


class FakeAdapterValue:
    def item(self):
        return 7

    def __format__(self, spec):
        return "trusted:" + spec


class FakeDataFrame:
    reset_index = object()

    def __init__(self):
        self.columns = ["date", "reset_index"]
        self.index = []
        self.items = {}

    def __setitem__(self, key, value):
        self.items[key] = value


class FakeSeries:
    def __init__(self):
        self.name = "before"
        self.index = []
        self.items = {}

    def __setitem__(self, key, value):
        self.items[key] = value


class FakeLoc:
    def __init__(self, obj):
        self.obj = obj
        self.items = {}

    def __setitem__(self, key, value):
        self.items[key] = value


class FakeILoc(FakeLoc):
    pass


PANDAS_TYPES = (FakeDataFrame, FakeSeries, (FakeLoc,), (FakeILoc,))


def host_adapters():
    scalar_types = (
        (np.bool_, bool),
        (np.int8, int),
        (np.int16, int),
        (np.int32, int),
        (np.int64, int),
        (np.uint8, int),
        (np.uint16, int),
        (np.uint32, int),
        (np.uint64, int),
        (np.float16, float),
        (np.float32, float),
        (np.float64, float),
    )
    adapters = []
    for host_type, native_type in scalar_types:
        if not any(host_type is existing[0] for existing in adapters):
            adapters.append((host_type, native_type, host_type.__format__, host_type.item))
    adapters.extend(
        (
            (pd.Timestamp, "timestamp", pd.Timestamp.__format__, None, datetime.datetime.year, datetime.datetime.tzinfo),
            (pd.Timedelta, "timedelta", pd.Timedelta.__format__, None),
        )
    )
    return tuple(adapters)


def format_with_adapters(template, *args):
    guarded_getattr = compat.make_compat_getattr(lambda obj, name, default=None: getattr(obj, name, default), host_adapters())
    return guarded_getattr(template, "format")(*args)


def format_with_fake_adapter(template, *args):
    adapter = (FakeAdapterValue, int, FakeAdapterValue.__format__, FakeAdapterValue.item)
    guarded_getattr = compat.make_compat_getattr(lambda obj, name, default=None: getattr(obj, name, default), (adapter,))
    return guarded_getattr(template, "format")(*args)


class TestRestrictedFormatting(unittest.TestCase):
    def assert_matches_str_format(self, template, *args, **kwargs):
        self.assertEqual(compat.compat_format(template, *args, **kwargs), template.format(*args, **kwargs))

    def test_supported_shapes_match_native_format(self):
        self.assert_matches_str_format("{} {name} {{ok}}", "first", name="named")
        self.assert_matches_str_format("{0} {name}", "first", name="named")
        self.assert_matches_str_format("{value:.2f} {percent:.3%}", value=1.234, percent=0.125)
        self.assert_matches_str_format("{0:>8d}", 42)
        self.assert_matches_str_format("{0}", ["a", 2, None])
        self.assert_matches_str_format("{0}", {"a": (1, False)})
        self.assert_matches_str_format("{0}", __import__("datetime").datetime(2026, 1, 2).strftime("%Y-%m-%d"))

    def test_repeated_fields_and_escaped_braces(self):
        self.assertEqual(compat.compat_format("{{{name}}} {name}", name="x"), "{x} x")

    def test_template_keyword_is_available_to_format_fields(self):
        self.assertEqual(compat.compat_format("{template}", template="ok"), "{template}".format(template="ok"))

    def test_format_map_arguments_are_positional_only(self):
        with self.assertRaises(TypeError):
            compat.compat_format_map(template="{name}", mapping={"name": "ok"})
        format_map = compat.compat_getattr(None, "{name}", "format_map")
        with self.assertRaises(TypeError):
            format_map(mapping={"name": "ok"})

    def test_missing_and_mixed_numbering_keep_stdlib_error_kinds(self):
        with self.assertRaises(IndexError):
            compat.compat_format("{}")
        with self.assertRaises(KeyError):
            compat.compat_format("{missing}")
        with self.assertRaises(ValueError):
            compat.compat_format("{} {0}", "x")
        with self.assertRaises(ValueError):
            compat.compat_format("{0} {}", "x")

    def test_format_map_requires_exact_dict_and_named_fields(self):
        self.assertEqual(compat.compat_format_map("{name}", {"name": "ok"}), "ok")
        with self.assertRaises(ValueError):
            compat.compat_format_map("{}", {"name": "ok"})
        with self.assertRaises(compat.FormatSecurityError):
            compat.compat_format_map("{name}", StringSubclass("not a dict"))
        with self.assertRaises(compat.FormatSecurityError):
            compat.compat_format_map("{name}", MappingBomb())

    def test_lookup_keys_are_exact_str_before_named_lookups(self):
        collision = CollisionKey()
        mapping = {collision: "unsafe"}
        CollisionKey.events.clear()
        with self.assertRaises(compat.FormatSecurityError):
            compat.compat_format_map("{name}", mapping)
        self.assertEqual(CollisionKey.events, [])

        with self.assertRaises(compat.FormatSecurityError):
            compat.compat_format_map("{name}", {StringSubclass("name"): "unsafe"})
        with self.assertRaises(compat.FormatSecurityError):
            compat.compat_format("{name}", **{StringSubclass("name"): "unsafe"})

    def test_unbound_str_format_is_delegated_to_the_restricted_getattr_guard(self):
        calls = []

        def safer_getattr(obj, name, default=None):
            calls.append((obj, name, default))
            raise AttributeError(name)

        with self.assertRaises(AttributeError):
            compat.compat_getattr(safer_getattr, str, "format")
        self.assertEqual(calls, [(str, "format", None)])

    def test_attribute_item_nested_specs_and_conversions_are_rejected(self):
        for template in ("{0.value}", "{0[key]}", "{0:{1}}", "{0!r}"):
            with self.subTest(template=template), self.assertRaises(compat.FormatSecurityError):
                compat.compat_format(template, "value", 3)

    def test_custom_objects_and_subclasses_never_run_hooks(self):
        with self.assertRaises(compat.FormatSecurityError):
            compat.compat_format("{0}", Bomb())
        with self.assertRaises(compat.FormatSecurityError):
            compat.compat_format("{0}", StringSubclass("unsafe"))

    def test_hostile_type_equality_never_reaches_format_hooks(self):
        EqualityBombMeta.events.clear()
        with self.assertRaises(compat.FormatSecurityError):
            compat.compat_format("{0}", EqualityBomb())
        self.assertEqual(EqualityBombMeta.events, [])

    def test_unused_unsafe_arguments_are_not_inspected(self):
        self.assertEqual(compat.compat_format("{0}", "safe", Bomb()), "safe")

    def test_budget_rejections(self):
        with self.assertRaises(compat.FormatSecurityError):
            compat.compat_format("{0:10001}", "x")
        with self.assertRaises(compat.FormatSecurityError):
            compat.compat_format("{0}", 1 << 4096)
        with self.assertRaises(compat.FormatSecurityError):
            compat.compat_format("{0}" * 4097, "x")

    def test_cyclic_container_is_rejected_without_rendering(self):
        value = []
        value.append(value)
        with self.assertRaises(compat.FormatSecurityError):
            compat.compat_format("{0}", value)

    def test_large_supported_presentations_match_native_format(self):
        cases = (
            ("{0:b}", (1 << 4096) - 1),
            ("{0:f}", 1e308),
            ("{0:.10000f}", 1),
            ("{0:.>10}", "x" * 20),
        )
        for template, value in cases:
            with self.subTest(template=template):
                self.assertEqual(compat.compat_format(template, value), template.format(value))

    def test_exact_temporal_types_match_native_supported_grammar(self):
        values = (
            datetime.datetime(2026, 1, 2, 3, 4, 5, 6789, datetime.timezone(datetime.timedelta(hours=3))),
            datetime.date(2026, 1, 2),
            datetime.time(3, 4, 5, 6789, datetime.timezone.utc),
        )
        for value in values:
            for spec in ("", "%Y%m%d%H%M%S%f%j%w%u%z%Z", "%% %Y-%m-%d"):
                with self.subTest(value=value, spec=spec):
                    self.assertEqual(compat.compat_format("{0:" + spec + "}", value), format(value, spec))

    def test_timedelta_and_decimal_match_native_contract(self):
        value = datetime.timedelta(days=-2, seconds=3, microseconds=4)
        self.assertEqual(compat.compat_format("{}", value), format(value, ""))
        with self.assertRaises(compat.FormatSecurityError):
            compat.compat_format("{:%s}", value)
        for value, spec in (
            (decimal.Decimal("1.2300"), ".2f"),
            (decimal.Decimal("1E+10000"), "g"),
            (decimal.Decimal("1E+10000"), ".10000f"),
        ):
            with self.subTest(value=value, spec=spec):
                self.assertEqual(compat.compat_format("{0:" + spec + "}", value), format(value, spec))

    def test_temporal_and_decimal_rejections_happen_before_rendering(self):
        calls = []

        class EvilTimezone(datetime.tzinfo):
            def utcoffset(self, value):
                calls.append("utcoffset")
                return datetime.timedelta()

            def dst(self, value):
                return datetime.timedelta()

            def tzname(self, value):
                calls.append("tzname")
                return "evil"

        with self.assertRaises(compat.FormatSecurityError):
            compat.compat_format("{}", datetime.datetime(2026, 1, 2, tzinfo=EvilTimezone()))
        self.assertEqual(calls, [])
        with patch.object(compat, "format", side_effect=AssertionError("must not render"), create=True):
            for spec in ("%Q", "%_d", "%Y" * 33):
                with self.subTest(spec=spec), self.assertRaises(compat.FormatSecurityError):
                    compat.compat_format("{0:" + spec + "}", datetime.datetime(2026, 1, 2))
        with self.assertRaises(compat.FormatSecurityError):
            compat.compat_format("{}", decimal.Decimal("NaN"))

    def test_temporal_and_decimal_subclasses_are_rejected(self):
        class DateSubclass(datetime.date):
            pass

        class DecimalSubclass(decimal.Decimal):
            pass

        with self.assertRaises(compat.FormatSecurityError):
            compat.compat_format("{}", DateSubclass(2026, 1, 2))
        with self.assertRaises(compat.FormatSecurityError):
            compat.compat_format("{}", DecimalSubclass("1"))

    def test_containers_reject_temporal_leaves_before_timezone_repr(self):
        value = datetime.datetime(2026, 1, 1, tzinfo=LocalTimezoneProbe())
        containers = ([value], (value,), {"value": value}, {value: "value"})
        for container in containers:
            LocalTimezoneProbe.events.clear()
            with self.subTest(container_type=type(container)), self.assertRaises(compat.FormatSecurityError):
                compat.compat_format("{}", container)
            self.assertEqual(LocalTimezoneProbe.events, [])

    def test_containers_reject_decimal_and_adapter_leaves_before_rendering(self):
        hidden_decimal = decimal.Decimal("9" * 4097)
        with patch.object(compat, "format", side_effect=AssertionError("must not render"), create=True):
            with self.assertRaises(compat.FormatSecurityError):
                compat.compat_format("{}", [hidden_decimal])

        self.assertEqual(format_with_fake_adapter("{}", FakeAdapterValue()), "trusted:")
        for container in ([FakeAdapterValue()], (FakeAdapterValue(),), {"value": FakeAdapterValue()}):
            with self.subTest(container_type=type(container)), self.assertRaises(compat.FormatSecurityError):
                format_with_fake_adapter("{}", container)

    def test_containers_reject_numpy_and_pandas_adapter_leaves(self):
        values = (np.int64(7), pd.Timestamp("2026-01-02"), pd.Timedelta("2 days"))
        for value in values:
            with self.subTest(value_type=type(value)):
                self.assertEqual(format_with_adapters("{}", value), format(value, ""))
                with self.assertRaises(compat.FormatSecurityError):
                    format_with_adapters("{}", [value])

    def test_exact_numpy_scalars_match_native_format_without_generic_item_support(self):
        integer_types = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
        values = [np.bool_(True)]
        for scalar_type in integer_types:
            info = np.iinfo(scalar_type)
            values.extend((scalar_type(info.min), scalar_type(info.max)))
        for scalar_type in (np.float16, np.float32, np.float64):
            values.extend((scalar_type("nan"), scalar_type("inf"), scalar_type("-inf"), scalar_type(1.25)))
        for value in values:
            specs = ("", ".2f", "s") if type(value) is np.bool_ else ("", ".2f")
            for spec in specs:
                with self.subTest(value_type=type(value), value=value, spec=spec):
                    try:
                        expected = format(value, spec)
                    except (TypeError, ValueError, OverflowError):
                        with self.assertRaises(compat.FormatSecurityError):
                            format_with_adapters("{0:" + spec + "}", value)
                    else:
                        self.assertEqual(format_with_adapters("{0:" + spec + "}", value), expected)

    def test_pandas_timestamp_and_timedelta_match_native_and_nat_is_rejected(self):
        timestamp = pd.Timestamp("2026-01-02 03:04:05.123456789+00:00")
        timedelta = pd.Timedelta("2 days 03:04:05.123456789")
        for value, spec in ((timestamp, ""), (timestamp, "%Y-%m-%d %H:%M:%S.%f"), (timedelta, "")):
            with self.subTest(value=value, spec=spec):
                self.assertEqual(format_with_adapters("{0:" + spec + "}", value), format(value, spec))
        with self.assertRaises(compat.FormatSecurityError):
            format_with_adapters("{}", pd.NaT)

    def test_cumulative_output_budget_rejects_before_the_next_field_renders(self):
        rendered_fields = 0
        builtin_format = format

        def render(value, spec):
            nonlocal rendered_fields
            rendered_fields += 1
            return builtin_format(value, spec)

        with patch.object(compat, "format", side_effect=render, create=True):
            with self.assertRaises(compat.FormatSecurityError):
                compat.compat_format("{0:10000}" * 105, "x")
        self.assertEqual(rendered_fields, 104)


class TestPandasWriteProxy(unittest.TestCase):
    def test_dataframe_only_permits_axes_and_existing_non_descriptor_columns(self):
        frame = FakeDataFrame()
        proxy = compat.PandasWriteProxy.for_target(frame, PANDAS_TYPES)

        proxy.date = [1]
        proxy.index = [0]
        proxy.columns = ["date", "reset_index"]
        proxy["new"] = [2]

        self.assertEqual(frame.items, {"date": [1], "new": [2]})
        self.assertEqual(frame.index, [0])
        self.assertEqual(frame.columns, ["date", "reset_index"])

        descriptor_frame = FakeDataFrame()
        descriptor_proxy = compat.PandasWriteProxy.for_target(descriptor_frame, PANDAS_TYPES)
        with self.assertRaises(TypeError):
            descriptor_proxy.reset_index = "blocked"
        self.assertEqual(descriptor_frame.columns, ["date", "reset_index"])
        self.assertEqual(descriptor_frame.items, {})
        with self.assertRaises(TypeError):
            proxy.new = [3]
        with self.assertRaises(AttributeError):
            proxy.items

    def test_series_and_indexer_permit_only_approved_writes(self):
        series = FakeSeries()
        series_proxy = compat.PandasWriteProxy.for_target(series, PANDAS_TYPES)
        series_proxy.name = "after"
        series_proxy[0] = 1
        self.assertEqual(series.name, "after")
        self.assertEqual(series.items, {0: 1})
        with self.assertRaises(TypeError):
            series_proxy.other = 1
        with self.assertRaises(TypeError):
            series_proxy._private = 1
        with self.assertRaises(TypeError):
            del series_proxy.name

        indexer = FakeLoc(series)
        indexer_proxy = compat.PandasWriteProxy.for_target(indexer, PANDAS_TYPES)
        indexer_proxy[0] = 2
        self.assertEqual(indexer.items, {0: 2})
        with self.assertRaises(TypeError):
            indexer_proxy.obj = series
        with self.assertRaises(TypeError):
            del indexer_proxy.obj

        with self.assertRaises(TypeError):
            compat.PandasWriteProxy.for_target(FakeLoc(object()), PANDAS_TYPES)

    def test_indexer_type_comparisons_do_not_call_hostile_equality(self):
        EqualityBombMeta.events.clear()

        class FakeIndexer(metaclass=EqualityBombMeta):
            pass

        self.assertIsNone(compat.PandasWriteProxy.for_target(FakeIndexer(), PANDAS_TYPES))
        self.assertEqual(EqualityBombMeta.events, [])

    def test_subclasses_are_rejected(self):
        class ChildDataFrame(FakeDataFrame):
            pass

        self.assertIsNone(compat.PandasWriteProxy.for_target(ChildDataFrame(), PANDAS_TYPES))


if __name__ == "__main__":
    unittest.main()
