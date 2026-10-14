"""Restricted-Python compatibility helpers with no Redash or pandas imports."""

import datetime
import decimal
import re
import string


class FormatSecurityError(ValueError):
    """Raised when a format operation exceeds the restricted compatibility contract."""


_MISSING = object()
_MAX_TEMPLATE_LENGTH = 64 * 1024
_MAX_FIELDS = 4096
_MAX_SPEC_LENGTH = 64
_MAX_WIDTH_PRECISION = 10000
_MAX_DEPTH = 16
_MAX_NODES = 10000
_MAX_STRINGS = 1024 * 1024
_MAX_OUTPUT = 1024 * 1024
_FORMAT_SPEC_RE = re.compile(
    r"(?:(?P<fill>.)?(?P<align>[<>=^]))?(?P<sign>[+\- ])?(?P<z>z)?"
    r"(?P<alternate>#)?(?P<zero>0)?(?P<width>[0-9]+)?(?P<grouping>[,_])?"
    r"(?:\.(?P<precision>[0-9]+))?(?P<type>[bcdeEfFgGnosxX%])?\Z"
)
_SAFE_CONTAINER_TYPES = (list, tuple, dict)
_TEMPORAL_DIRECTIVES = frozenset("YmdHMSfjwuzZ")
_TEMPORAL_DIRECTIVE_BOUNDS = {
    "Y": 10,
    "m": 2,
    "d": 2,
    "H": 2,
    "M": 2,
    "S": 2,
    "f": 6,
    "j": 3,
    "w": 1,
    "u": 1,
    "z": 16,
    "Z": 256,
}


def _raise_security_error(message):
    raise FormatSecurityError(message)


def _is_exact_type(value_type, allowed_types):
    return any(value_type is allowed_type for allowed_type in allowed_types)


def _validate_spec(spec):
    if len(spec) > _MAX_SPEC_LENGTH or "{" in spec or "}" in spec:
        _raise_security_error("unsupported format specification")

    match = _FORMAT_SPEC_RE.fullmatch(spec)
    if match is None:
        _raise_security_error("unsupported format specification")

    for group in ("width", "precision"):
        value = match.group(group)
        if value is not None and int(value) > _MAX_WIDTH_PRECISION:
            _raise_security_error("format width or precision exceeds the limit")
    precision = match.group("precision")
    return int(match.group("width") or 0), int(precision or 0), precision is not None, match


def _validate_temporal_spec(spec):
    if len(spec) > _MAX_SPEC_LENGTH or "{" in spec or "}" in spec:
        _raise_security_error("unsupported format specification")

    bound = 0
    index = 0
    while index < len(spec):
        if spec[index] != "%":
            bound += 1
            index += 1
            continue
        index += 1
        if index == len(spec):
            _raise_security_error("unsupported format specification")
        directive = spec[index]
        index += 1
        if directive == "%":
            bound += 1
        elif directive in _TEMPORAL_DIRECTIVES:
            bound += _TEMPORAL_DIRECTIVE_BOUNDS[directive]
        else:
            _raise_security_error("unsupported format specification")
    return bound


def _validate_timezone(value, tzinfo_descriptor):
    tzinfo = tzinfo_descriptor.__get__(value, type(value))
    if tzinfo is None:
        return
    if type(tzinfo) is not datetime.timezone:
        _raise_security_error("unsupported timezone")
    name = datetime.timezone.tzname(tzinfo, None)
    if type(name) is not str or len(name) > 256:
        _raise_security_error("unsupported timezone")


def _validate_datetime(value):
    _validate_timezone(value, datetime.datetime.tzinfo)


def _validate_time(value):
    _validate_timezone(value, datetime.time.tzinfo)


def _validate_decimal(value):
    if decimal.Decimal.__sizeof__(value) > 8 * 1024 or not decimal.Decimal.is_finite(value):
        _raise_security_error("unsupported decimal value")
    value_tuple = decimal.Decimal.as_tuple(value)
    if len(value_tuple.digits) > 4096 or abs(value_tuple.exponent) > 10000:
        _raise_security_error("decimal value exceeds the limit")
    return len(value_tuple.digits), abs(value_tuple.exponent)


def _find_adapter(value_type, adapters):
    for adapter in adapters:
        if value_type is adapter[0]:
            return adapter
    return None


def _estimate_string_repr(value):
    # A conservative upper bound for repr(str) without creating that representation.
    return 2 + sum(2 if char in "\\'" else 1 if 32 <= ord(char) < 127 else 10 for char in value)


class _ValueValidator:
    def __init__(self, template_length):
        self.nodes = 0
        self.string_size = template_length

    def validate(self, value, adapters=()):
        return self._validate(value, 0, set(), adapters)

    def _validate(self, value, depth, ancestors, adapters):
        if depth > _MAX_DEPTH:
            _raise_security_error("format value nesting exceeds the limit")

        self.nodes += 1
        if self.nodes > _MAX_NODES:
            _raise_security_error("format value node count exceeds the limit")

        value_type = type(value)
        if value_type is str:
            self.string_size += len(value)
            if self.string_size > _MAX_STRINGS:
                _raise_security_error("format string input exceeds the limit")
            return _estimate_string_repr(value)
        if value_type is int:
            if value.bit_length() > 4096:
                _raise_security_error("format integer exceeds the limit")
            return max(1, (value.bit_length() * 30103) // 100000 + 2)
        if value_type is float:
            return len(repr(value))
        if value_type is bool:
            return 4 if value else 5
        if value is None:
            return 4
        if (
            value_type is datetime.datetime
            or value_type is datetime.date
            or value_type is datetime.time
            or value_type is datetime.timedelta
            or value_type is decimal.Decimal
        ):
            if depth:
                _raise_security_error("containers only support builtin primitive leaves")
            return 64
        adapter = _find_adapter(value_type, adapters)
        if adapter is not None:
            if depth:
                _raise_security_error("containers only support builtin primitive leaves")
            return 64
        if not _is_exact_type(value_type, _SAFE_CONTAINER_TYPES):
            _raise_security_error("unsupported format value type")

        value_id = id(value)
        if value_id in ancestors:
            _raise_security_error("cyclic format value")
        ancestors.add(value_id)
        try:
            if value_type is list:
                estimate = 2 + sum(self._validate(item, depth + 1, ancestors, adapters) for item in value)
                estimate += max(0, len(value) - 1) * 2
            elif value_type is tuple:
                estimate = 2 + sum(self._validate(item, depth + 1, ancestors, adapters) for item in value)
                estimate += max(0, len(value) - 1) * 2 + (1 if len(value) == 1 else 0)
            else:
                estimate = 2
                for index, (key, item) in enumerate(value.items()):
                    if index:
                        estimate += 2
                    estimate += self._validate(key, depth + 1, ancestors, adapters)
                    estimate += 2
                    estimate += self._validate(item, depth + 1, ancestors, adapters)
        finally:
            ancestors.remove(value_id)

        if estimate > _MAX_OUTPUT:
            _raise_security_error("format container representation exceeds the limit")
        return estimate


def _field_kind(field_name):
    if field_name == "":
        return "automatic"
    if field_name.isascii() and field_name.isdecimal():
        return "positional"
    if field_name.isidentifier():
        return "named"
    _raise_security_error("attribute and item traversal are not permitted in format fields")


def _numeric_output_bound(value, estimate, precision, has_precision, match):
    value_type = type(value)
    presentation = match.group("type")
    uses_float_presentation = presentation is not None and presentation in "eEfFgG%"

    if (value_type is float or value_type is int or value_type is bool) and (
        uses_float_presentation or (value_type is float and has_precision)
    ):
        # Fixed-point formatting of finite doubles needs up to 309 integral digits;
        # leave room for fractional digits, signs, decimal points, and exponent forms.
        return (precision if has_precision else 6) + 330

    if value_type is int or value_type is bool:
        bit_length = value.bit_length()
        if presentation == "b":
            return max(1, bit_length)
        if presentation == "o":
            return max(1, (bit_length + 2) // 3)
        if presentation == "x" or presentation == "X":
            return max(1, (bit_length + 3) // 4)
        if presentation == "c":
            return 1
        return estimate

    return estimate


def _output_bound(value, estimate, width, precision, has_precision, match):
    value_type = type(value)
    if _is_exact_type(value_type, _SAFE_CONTAINER_TYPES):
        return estimate
    if value_type is str:
        content = min(len(value), precision) if has_precision else len(value)
        return max(width, content)
    content = _numeric_output_bound(value, estimate, precision, has_precision, match)
    prefix = 2 if match.group("alternate") is not None else 0
    grouping = (content - 1) // 3 if match.group("grouping") is not None else 0
    # A value may be negative even when no explicit sign option was requested.
    return max(width, content + 1 + prefix + grouping)


def _adapter_output_bound(value, adapter, width, precision, has_precision, match):
    native_value = adapter[3](value)
    if type(native_value) is not adapter[1]:
        _raise_security_error("unsupported format value type")
    estimate = _ValueValidator(0).validate(native_value)
    return _output_bound(native_value, estimate, width, precision, has_precision, match)


def _decimal_output_bound(value, width, precision, has_precision):
    digits, exponent = _validate_decimal(value)
    content = 2 * (digits + exponent + (precision if has_precision else 0) + 32)
    return max(width, content)


def _temporal_output_bound(value, value_type, spec, adapter):
    if value_type is datetime.datetime:
        _validate_datetime(value)
    elif value_type is datetime.time:
        _validate_time(value)
    elif adapter is not None and adapter[1] == "timestamp":
        year = adapter[4].__get__(value, value_type)
        if not 1 <= year <= 9999:
            _raise_security_error("unsupported timestamp")
        _validate_timezone(value, adapter[5])

    if not spec:
        return 64
    return _validate_temporal_spec(spec)


def _render_value(value, spec, adapter):
    if adapter is not None:
        return adapter[2](value, spec)
    return format(value, spec)


def _validate_lookup_keys(lookup):
    for key in lookup:
        if type(key) is not str:
            _raise_security_error("format lookup keys must be exact str")


def _render(template, args, kwargs, mapping=_MISSING, adapters=()):
    if type(template) is not str:
        _raise_security_error("format receiver must be an exact str")
    if len(template) > _MAX_TEMPLATE_LENGTH:
        _raise_security_error("format template exceeds the limit")

    _validate_lookup_keys(mapping if mapping is not _MISSING else kwargs)

    formatter = string.Formatter()
    pieces = []
    validator = _ValueValidator(len(template))
    automatic_index = 0
    numbering = None
    field_count = 0

    try:
        parsed = tuple(formatter.parse(template))
    except ValueError:
        _raise_security_error("invalid format template")

    output_length = 0
    for literal, field_name, spec, conversion in parsed:
        output_length += len(literal)
        if output_length > _MAX_OUTPUT:
            _raise_security_error("formatted output exceeds the limit")
        pieces.append(literal)
        if field_name is None:
            continue

        field_count += 1
        if field_count > _MAX_FIELDS:
            _raise_security_error("format field count exceeds the limit")
        if conversion is not None:
            _raise_security_error("format conversions are not permitted")
        kind = _field_kind(field_name)
        if mapping is not _MISSING and kind != "named":
            raise ValueError("Format string contains positional fields")
        if kind == "automatic":
            if numbering == "positional":
                raise ValueError("cannot switch from manual field specification to automatic field numbering")
            numbering = "automatic"
            try:
                value = args[automatic_index]
            except IndexError:
                raise
            automatic_index += 1
        elif kind == "positional":
            if numbering == "automatic":
                raise ValueError("cannot switch from automatic field numbering to manual field specification")
            numbering = "positional"
            value = args[int(field_name)]
        elif mapping is not _MISSING:
            value = mapping[field_name]
        else:
            value = kwargs[field_name]

        value_type = type(value)
        adapter = _find_adapter(value_type, adapters)
        is_temporal = (
            value_type is datetime.datetime
            or value_type is datetime.date
            or value_type is datetime.time
            or value_type is datetime.timedelta
            or (adapter is not None and adapter[1] in ("timestamp", "timedelta"))
        )
        if is_temporal:
            if value_type is datetime.timedelta or (adapter is not None and adapter[1] == "timedelta"):
                if spec:
                    _raise_security_error("timedelta only supports an empty format specification")
            field_bound = _temporal_output_bound(value, value_type, spec, adapter)
        else:
            width, precision, has_precision, match = _validate_spec(spec)
            if value_type is decimal.Decimal:
                field_bound = _decimal_output_bound(value, width, precision, has_precision)
            elif adapter is not None:
                field_bound = _adapter_output_bound(value, adapter, width, precision, has_precision, match)
            else:
                estimate = validator.validate(value, adapters)
                if _is_exact_type(value_type, _SAFE_CONTAINER_TYPES) and spec:
                    _raise_security_error("containers only support an empty format specification")
                field_bound = _output_bound(value, estimate, width, precision, has_precision, match)
        if is_temporal:
            # Keep the scalar allow-list closed before any renderer is reached.
            validator.validate(value, adapters)
        if field_bound > _MAX_OUTPUT - output_length:
            _raise_security_error("format value exceeds output limit")
        try:
            rendered = _render_value(value, spec, adapter)
        except (TypeError, ValueError, OverflowError):
            _raise_security_error("format specification is incompatible with value type")
        pieces.append(rendered)
        output_length += len(rendered)
        if output_length > _MAX_OUTPUT:
            _raise_security_error("formatted output exceeds the limit")

    result = "".join(pieces)
    if len(result) > _MAX_OUTPUT:
        _raise_security_error("formatted output exceeds the limit")
    return result


def compat_format(template, /, *args, **kwargs):
    return _render(template, args, kwargs)


def compat_format_map(template, mapping, /):
    if type(mapping) is not dict:
        _raise_security_error("format_map requires an exact dict")
    return _render(template, (), {}, mapping=mapping)


def _compat_getattr(safer_getattr, obj, name, default, adapters):
    """Expose controlled formatting methods while preserving safer_getattr semantics."""
    if type(obj) is str and type(name) is str:
        if name == "format":
            return lambda *args, **kwargs: _render(obj, args, kwargs, adapters=adapters)
        if name == "format_map":
            return lambda mapping, /: _render(obj, (), {}, mapping=mapping, adapters=adapters)
    if default is _MISSING:
        return safer_getattr(obj, name)
    return safer_getattr(obj, name, default)


def compat_getattr(safer_getattr, obj, name, default=_MISSING):
    """Default stdlib-only restricted attribute guard."""
    return _compat_getattr(safer_getattr, obj, name, default, ())


def make_compat_getattr(safer_getattr, adapters):
    """Create a guard with host-provided exact scalar adapters."""
    adapters = tuple(adapters)

    def guarded_getattr(obj, name, default=_MISSING):
        return _compat_getattr(safer_getattr, obj, name, default, adapters)

    return guarded_getattr


class PandasWriteProxy:
    """Write-only proxy for exact pandas public objects; it never forwards reads."""

    __slots__ = ("_pandas_target", "_pandas_kind")

    def __init__(self, target, kind):
        object.__setattr__(self, "_pandas_target", target)
        object.__setattr__(self, "_pandas_kind", kind)

    @classmethod
    def for_target(cls, target, types):
        if types is None:
            return None
        dataframe_type, series_type, loc_types, iloc_types = types
        target_type = type(target)
        if target_type is dataframe_type:
            return cls(target, "dataframe")
        if target_type is series_type:
            return cls(target, "series")
        if _is_exact_type(target_type, loc_types) or _is_exact_type(target_type, iloc_types):
            owner = target.obj
            if type(owner) is dataframe_type or type(owner) is series_type:
                return cls(target, "indexer")
            raise TypeError("pandas indexer owner is not allowed")
        return None

    def __setitem__(self, key, value):
        object.__getattribute__(self, "_pandas_target")[key] = value

    def __setattr__(self, name, value):
        target = object.__getattribute__(self, "_pandas_target")
        kind = object.__getattribute__(self, "_pandas_kind")
        if type(name) is not str or name.startswith("_"):
            raise TypeError("pandas private attributes cannot be modified")
        if kind == "indexer":
            raise TypeError("pandas indexer attributes cannot be modified")
        if kind == "dataframe":
            if name in ("columns", "index"):
                setattr(target, name, value)
                return
            if name in target.columns and not any(name in parent.__dict__ for parent in type(target).__mro__):
                target[name] = value
                return
            raise TypeError("pandas dataframe attribute cannot be modified")
        if name in ("name", "index"):
            setattr(target, name, value)
            return
        raise TypeError("pandas series attribute cannot be modified")

    def __delattr__(self, name):
        raise TypeError("pandas attributes cannot be deleted")
