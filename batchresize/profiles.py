"""Validated, portable task profiles.

Profiles never contain input/output paths, commands, expressions or tokens.
The module is standard-library only so the GUI and the CLI can share exactly the
same validator without importing Pillow, NumPy or CUDA.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from .variants import OutputVariant, VariantError

PROFILE_SCHEMA_VERSION = 1
PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class ProfileError(ValueError):
    """A profile is malformed, unsupported or not safe to import."""


def _fail(path: str, message: str) -> None:
    raise ProfileError(f"{path}: {message}")


def _mapping(value, path: str, allowed: set[str]) -> dict:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    unknown = set(value) - allowed
    if unknown:
        _fail(path, "unknown field(s): " + ", ".join(sorted(unknown)))
    return value


def _text(value, path: str, *, maximum: int = 120) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a non-empty string")
    text = value.strip()
    if len(text) > maximum:
        _fail(path, f"must be at most {maximum} characters")
    return text


def _bool(value, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "must be true or false")
    return value


def _integer(value, path: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "must be an integer")
    if not minimum <= value <= maximum:
        _fail(path, f"must be between {minimum} and {maximum}")
    return value


def _number(value, path: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a number")
    number = float(value)
    if not minimum <= number <= maximum:
        _fail(path, f"must be between {minimum:g} and {maximum:g}")
    return number


def _hex_colour(value, path: str) -> str:
    text = _text(value, path, maximum=7)
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", text):
        _fail(path, "must be a #RRGGBB colour")
    return text.upper()


def parse_colour(value: str) -> tuple[int, int, int]:
    """Canonical #RRGGBB parser shared by profile validation and the GUI."""
    text = _hex_colour(value, "colour").lstrip("#")
    return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _validate_size(value) -> dict:
    data = _mapping(value, "size", {"mode", "width", "height", "value"})
    if "mode" not in data:
        _fail("size.mode", "is required")
    mode = data["mode"]
    if mode not in ("resolution", "percent", "megapixels", "longest", "shortest"):
        _fail("size.mode", "must be resolution, percent, megapixels, longest or shortest")
    if mode == "resolution":
        width = data.get("width")
        height = data.get("height")
        if width is None or height is None:
            _fail("size", "resolution mode requires width and height")
        width = _integer(width, "size.width", 1, 100_000)
        height = _integer(height, "size.height", 1, 100_000)
        if data.get("value") not in (None, 0):
            _fail("size.value", "must be null for resolution mode")
        return {"mode": mode, "width": width, "height": height, "value": None}
    if "value" not in data:
        _fail("size.value", "is required")
    value_number = _number(data["value"], "size.value", 1e-6, 100_000)
    return {"mode": mode, "width": None, "height": None, "value": value_number}


def _validate_fit(value) -> dict:
    data = _mapping(value, "fit", {"mode", "allow_upscale", "crop_position"})
    mode = data.get("mode", "contain")
    if mode not in ("contain", "cover", "pad"):
        _fail("fit.mode", "must be contain, cover or pad")
    if mode == "pad":
        _fail("fit.mode", "'pad' is not supported by this version")
    position = data.get("crop_position", "center")
    if position != "center":
        _fail("fit.crop_position", "only 'center' is supported")
    return {"mode": mode,
            "allow_upscale": _bool(data.get("allow_upscale", False),
                                   "fit.allow_upscale"),
            "crop_position": position}


def _validate_sharpen(value) -> dict:
    data = _mapping(value, "sharpen", {"enabled", "amount"})
    enabled = _bool(data.get("enabled", False), "sharpen.enabled")
    amount = _number(data.get("amount", 0.0), "sharpen.amount", 0.0, 5.0)
    if enabled:
        _fail("sharpen.enabled", "output sharpening is not supported by this version")
    return {"enabled": False, "amount": amount}


def _validate_colour(value) -> dict:
    data = _mapping(value, "color", {"policy"})
    policy = data.get("policy", "preserve")
    if policy != "preserve":
        _fail("color.policy", "only 'preserve' is supported by this version")
    return {"policy": policy}


def _validate_jpeg(value) -> dict:
    data = _mapping(value, "jpeg", {"quality", "subsampling", "bg_color"})
    quality = _integer(data.get("quality", 80), "jpeg.quality", 1, 100)
    subsampling = data.get("subsampling", "auto")
    if subsampling not in ("auto", "444", "422", "420"):
        _fail("jpeg.subsampling", "must be auto, 444, 422 or 420")
    bg = _hex_colour(data.get("bg_color", "#FFFFFF"), "jpeg.bg_color")
    return {"quality": quality, "subsampling": subsampling, "bg_color": bg}


def _validate_metadata(value) -> dict:
    data = _mapping(value, "metadata", {"exif", "icc"})
    return {"exif": _bool(data.get("exif", True), "metadata.exif"),
            "icc": _bool(data.get("icc", True), "metadata.icc")}


def _validate_variants(value) -> tuple[dict, ...]:
    if not isinstance(value, list):
        _fail("variants", "must be a list")
    result: list[dict] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        path = f"variants[{index}]"
        data = _mapping(item, path, {"id", "name", "size", "fit", "jpeg"})
        if "id" not in data:
            _fail(f"{path}.id", "is required")
        variant_id = _text(data["id"], f"{path}.id", maximum=64)
        if not PROFILE_ID_RE.fullmatch(variant_id):
            _fail(f"{path}.id", "must match [a-z0-9][a-z0-9._-]{0,63}")
        if variant_id in seen:
            _fail(f"{path}.id", "duplicate variant id")
        seen.add(variant_id)
        try:
            output = OutputVariant.from_profile_dict(data)
        except VariantError as exc:
            raise ProfileError(f"{path}: {exc}") from exc
        entry = output.to_profile_dict()
        entry["name"] = _text(data.get("name", entry["name"]),
                              f"{path}.name", maximum=80)
        result.append(entry)
    return tuple(result)


def migrate_profile(data: dict) -> dict:
    """Return a version-1 profile document.

    Every future schema bump adds a numbered migration here. A profile from a
    newer writer is rejected instead of being interpreted with missing fields.
    """
    if not isinstance(data, dict):
        _fail("profile", "must be an object")
    version = data.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        _fail("schema_version", "is required and must be an integer")
    if version > PROFILE_SCHEMA_VERSION:
        _fail("schema_version",
              f"version {version} is newer than the supported "
              f"version {PROFILE_SCHEMA_VERSION}")
    if version < 1:
        _fail("schema_version", f"unsupported version {version}")
    return copy.deepcopy(data)


def parse_profile(data: dict, *, builtin: bool = False) -> "Profile":
    data = migrate_profile(data)
    _mapping(data, "profile",
             {"schema_version", "id", "name", "description", "size", "fit",
              "filter", "sharpen", "color", "format", "jpeg", "metadata",
              "variants"})
    profile_id = _text(data.get("id"), "id", maximum=64)
    if not PROFILE_ID_RE.fullmatch(profile_id):
        _fail("id", "must match [a-z0-9][a-z0-9._-]{0,63}")
    name = _text(data.get("name"), "name", maximum=80)
    description = data.get("description", "")
    if not isinstance(description, str) or len(description) > 400:
        _fail("description", "must be a string of at most 400 characters")
    if "size" not in data:
        _fail("size", "is required")
    if "jpeg" not in data:
        _fail("jpeg", "is required")
    filter_name = data.get("filter", "lanczos3")
    if filter_name not in ("box", "bilinear", "lanczos3"):
        _fail("filter", "must be box, bilinear or lanczos3")
    output_format = data.get("format", "jpeg")
    if output_format != "jpeg":
        _fail("format", "only JPEG output is supported by this version")
    return Profile(
        schema_version=PROFILE_SCHEMA_VERSION,
        id=profile_id,
        name=name,
        description=description.strip(),
        size=_validate_size(data["size"]),
        fit=_validate_fit(data.get("fit", {"mode": "contain"})),
        filter=filter_name,
        sharpen=_validate_sharpen(data.get("sharpen", {"enabled": False})),
        color=_validate_colour(data.get("color", {"policy": "preserve"})),
        format=output_format,
        jpeg=_validate_jpeg(data["jpeg"]),
        metadata=_validate_metadata(data.get("metadata", {})),
        variants=_validate_variants(data.get("variants", [])),
        builtin=builtin,
    )


@dataclass(frozen=True)
class Profile:
    schema_version: int
    id: str
    name: str
    description: str
    size: dict
    fit: dict
    filter: str
    sharpen: dict
    color: dict
    format: str
    jpeg: dict
    metadata: dict
    variants: tuple[dict, ...] = ()
    builtin: bool = False

    @property
    def sizing(self) -> str:
        return str(self.size["mode"])

    @property
    def width(self) -> int | None:
        return self.size.get("width")

    @property
    def height(self) -> int | None:
        return self.size.get("height")

    @property
    def sizing_value(self) -> float:
        return float(self.size.get("value") or 0.0)

    @property
    def mode(self) -> str:
        return str(self.fit["mode"])

    @property
    def allow_upscale(self) -> bool:
        return bool(self.fit["allow_upscale"])

    @property
    def quality(self) -> int:
        return int(self.jpeg["quality"])

    @property
    def subsampling(self) -> str:
        return str(self.jpeg["subsampling"])

    @property
    def bg_color(self) -> str:
        return str(self.jpeg["bg_color"])

    @property
    def keep_exif(self) -> bool:
        return bool(self.metadata["exif"])

    @property
    def keep_icc(self) -> bool:
        return bool(self.metadata["icc"])

    def to_dict(self) -> dict:
        data = {
            "schema_version": self.schema_version,
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "size": copy.deepcopy(self.size),
            "fit": copy.deepcopy(self.fit),
            "filter": self.filter,
            "sharpen": copy.deepcopy(self.sharpen),
            "color": copy.deepcopy(self.color),
            "format": self.format,
            "jpeg": copy.deepcopy(self.jpeg),
            "metadata": copy.deepcopy(self.metadata),
            "variants": copy.deepcopy(list(self.variants)),
        }
        return data


def _builtin(profile_id: str, name: str, description: str, *, size: dict,
             mode: str, quality: int, subsampling: str,
             allow_upscale: bool = False, filter_name: str = "lanczos3",
             bg_color: str = "#FFFFFF") -> Profile:
    return parse_profile({
        "schema_version": PROFILE_SCHEMA_VERSION,
        "id": profile_id,
        "name": name,
        "description": description,
        "size": size,
        "fit": {"mode": mode, "allow_upscale": allow_upscale,
                "crop_position": "center"},
        "filter": filter_name,
        "sharpen": {"enabled": False, "amount": 0.0},
        "color": {"policy": "preserve"},
        "format": "jpeg",
        "jpeg": {"quality": quality, "subsampling": subsampling,
                 "bg_color": bg_color},
        "metadata": {"exif": True, "icc": True},
        "variants": [],
    }, builtin=True)


def builtin_profiles() -> list[Profile]:
    return [
        _builtin("builtin-web-photo", "Фото для сайта",
                 "Вписывание в 1600 px с сохранением пропорций и метаданных.",
                 size={"mode": "resolution", "width": 1600, "height": 1600,
                       "value": None},
                 mode="contain", quality=85, subsampling="420"),
        _builtin("builtin-thumbnails", "Миниатюры",
                 "Квадратная миниатюра 320 px с центральной обрезкой.",
                 size={"mode": "resolution", "width": 320, "height": 320,
                       "value": None},
                 mode="cover", quality=80, subsampling="420"),
        _builtin("builtin-product-cards", "Карточки товаров",
                 "Квадратное изображение 1000 px с белым фоном.",
                 size={"mode": "resolution", "width": 1000, "height": 1000,
                       "value": None},
                 mode="cover", quality=85, subsampling="420"),
        _builtin("builtin-screenshots-text", "Скриншоты и текст",
                 "Бережное сжатие без хромы-субдискретизации для мелкого текста.",
                 size={"mode": "resolution", "width": 1920, "height": 1920,
                       "value": None},
                 mode="contain", quality=92, subsampling="444",
                 filter_name="bilinear"),
    ]


def default_profiles_path() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return root / "batchresize" / "profiles.json"
    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "batchresize" / "profiles.json"


def _collection(profiles: Iterable[Profile]) -> dict:
    return {"schema_version": PROFILE_SCHEMA_VERSION,
            "profiles": [profile.to_dict() for profile in profiles]}


def parse_collection(data: dict, *, source: str = "profiles") -> list[Profile]:
    if not isinstance(data, dict):
        _fail(source, "must be an object")
    unknown = set(data) - {"schema_version", "profiles"}
    if unknown:
        _fail(source, "unknown field(s): " + ", ".join(sorted(unknown)))
    version = data.get("schema_version")
    if version != PROFILE_SCHEMA_VERSION:
        _fail(f"{source}.schema_version",
              f"must be {PROFILE_SCHEMA_VERSION} (got {version!r})")
    entries = data.get("profiles")
    if not isinstance(entries, list):
        _fail(f"{source}.profiles", "must be a list")
    result: list[Profile] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        try:
            profile = parse_profile(entry)
        except ProfileError as exc:
            raise ProfileError(f"{source}.profiles[{index}]: {exc}") from exc
        if profile.id in seen:
            _fail(f"{source}.profiles[{index}].id", "duplicate profile id")
        seen.add(profile.id)
        result.append(profile)
    return result


def load_profiles(path: Path, *, strict: bool = False) -> tuple[list[Profile], list[str]]:
    """Load a profile store. A damaged store never prevents the app from starting."""
    path = Path(path)
    if not path.is_file():
        return [], []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return parse_collection(raw, source=str(path)), []
    except (OSError, ValueError, ProfileError) as exc:
        if strict:
            raise
        return [], [f"profiles: {path}: {exc}"]


def _atomic_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    handle = tempfile.NamedTemporaryFile(prefix=path.name + ".",
                                         suffix=".tmp", dir=str(path.parent),
                                         delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def save_profiles(path: Path, profiles: Iterable[Profile]) -> None:
    entries = list(profiles)
    seen: set[str] = set()
    for profile in entries:
        # Re-parse the serialised form: the save path must never bypass validation.
        parse_profile(profile.to_dict())
        if profile.id in seen:
            raise ProfileError(f"duplicate profile id: {profile.id}")
        seen.add(profile.id)
    _atomic_json(Path(path), _collection(entries))


def read_profile_file(path: Path) -> list[Profile]:
    """Import only data, accepting one profile or a collection."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProfileError(f"{path}: {exc}") from exc
    if isinstance(raw, dict) and "profiles" in raw:
        return parse_collection(raw, source=str(path))
    return [parse_profile(raw)]


def export_profile(path: Path, profile: Profile) -> None:
    _atomic_json(Path(path), profile.to_dict())


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:48] or "profile"


def unique_id(profiles: Iterable[Profile], base: str) -> str:
    used = {profile.id for profile in profiles}
    candidate = _slug(base)
    if candidate not in used and PROFILE_ID_RE.fullmatch(candidate):
        return candidate
    while True:
        candidate = f"{_slug(base)}-{uuid.uuid4().hex[:8]}"
        if candidate not in used:
            return candidate


def unique_name(profiles: Iterable[Profile], base: str) -> str:
    used = {profile.name.casefold() for profile in profiles}
    if base.casefold() not in used:
        return base
    index = 2
    while f"{base} ({index})".casefold() in used:
        index += 1
    return f"{base} ({index})"


def build_profile(name: str, *, size: dict, mode: str, allow_upscale: bool,
                  filter_name: str, quality: int, subsampling: str,
                  bg_color: str, keep_exif: bool, keep_icc: bool,
                  profile_id: str | None = None,
                  description: str = "", variants=()) -> Profile:
    return parse_profile({
        "schema_version": PROFILE_SCHEMA_VERSION,
        "id": profile_id or _slug(name),
        "name": name,
        "description": description,
        "size": size,
        "fit": {"mode": mode, "allow_upscale": allow_upscale,
                "crop_position": "center"},
        "filter": filter_name,
        "sharpen": {"enabled": False, "amount": 0.0},
        "color": {"policy": "preserve"},
        "format": "jpeg",
        "jpeg": {"quality": quality, "subsampling": subsampling,
                 "bg_color": bg_color},
        "metadata": {"exif": keep_exif, "icc": keep_icc},
        "variants": list(variants),
    })


def find_profile(profiles: Iterable[Profile], key: str) -> Profile | None:
    key_folded = key.casefold()
    for profile in profiles:
        if profile.id.casefold() == key_folded or profile.name.casefold() == key_folded:
            return profile
    return None


def import_into(existing: list[Profile], imported: Iterable[Profile]) -> list[Profile]:
    """Append imported profiles, renaming id/name collisions instead of overwriting."""
    result = list(existing)
    for profile in imported:
        if find_profile(result, profile.id) is not None:
            profile = replace(profile, id=unique_id(result, profile.id))
        if find_profile(result, profile.name) is not None:
            profile = replace(profile, name=unique_name(result, profile.name))
        result.append(profile)
    return result
