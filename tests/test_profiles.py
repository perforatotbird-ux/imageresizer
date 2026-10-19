"""Task profiles: validation, portability, storage and CLI precedence."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from batchresize import cli, profiles


def builtin(profile_id: str) -> profiles.Profile:
    return next(item for item in profiles.builtin_profiles() if item.id == profile_id)


def test_builtin_profiles_are_valid_and_distinct():
    items = profiles.builtin_profiles()
    assert len(items) == 4
    assert len({item.id for item in items}) == 4
    assert all(item.builtin for item in items)
    assert all(item.schema_version == profiles.PROFILE_SCHEMA_VERSION for item in items)
    names = {item.name for item in items}
    assert {"Фото для сайта", "Миниатюры", "Карточки товаров",
            "Скриншоты и текст"} <= names


def test_export_import_preserves_every_parameter(tmp_path):
    original = builtin("builtin-product-cards")
    path = tmp_path / "profile.json"
    profiles.export_profile(path, original)
    imported = profiles.read_profile_file(path)
    assert len(imported) == 1
    assert imported[0].to_dict() == original.to_dict()


def test_profile_contains_no_paths_or_executable_fields():
    data = builtin("builtin-web-photo").to_dict()
    assert set(data) == {
        "schema_version", "id", "name", "description", "size", "fit",
        "filter", "sharpen", "color", "format", "jpeg", "metadata", "variants"}
    text = repr(data).lower()
    assert "input" not in text
    assert "output" not in text
    assert "command" not in text
    assert "eval" not in text


def test_unknown_fields_are_rejected_with_a_path():
    data = builtin("builtin-thumbnails").to_dict()
    data["command"] = "del /s /q"
    with pytest.raises(profiles.ProfileError, match="unknown field"):
        profiles.parse_profile(data)


@pytest.mark.parametrize("path,value,message", [
    (("fit", "mode"), "pad", "not supported"),
    (("sharpen", "amount"), -1.0, "between 0.0 and 10.0"),
    (("color", "policy"), "srgb", "only 'preserve'"),
    (("format",), "png", "only JPEG"),
    (("filter",), "unsupported_filter", "must be one of"),
    (("jpeg", "quality"), 0, "between 1 and 100"),
    (("jpeg", "subsampling"), "411", "must be auto"),
])
def test_unsupported_values_are_rejected(path, value, message):
    data = builtin("builtin-web-photo").to_dict()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(profiles.ProfileError, match=message):
        profiles.parse_profile(data)


def test_stage4_values_are_accepted():
    data = builtin("builtin-web-photo").to_dict()
    data["filter"] = "nearest"
    data["sharpen"] = {"enabled": True, "preset": "standard", "amount": 1.2, "radius": 1.0, "threshold": 2}
    profile = profiles.parse_profile(data)
    assert profile.filter == "nearest"
    assert profile.sharpen["enabled"] is True


def test_resolution_requires_both_dimensions():
    data = builtin("builtin-web-photo").to_dict()
    data["size"]["height"] = None
    with pytest.raises(profiles.ProfileError, match="requires width and height"):
        profiles.parse_profile(data)


@pytest.mark.parametrize("version", [None, 0, 2])
def test_unknown_schema_versions_are_rejected(version):
    data = builtin("builtin-web-photo").to_dict()
    if version is None:
        data.pop("schema_version")
    else:
        data["schema_version"] = version
    with pytest.raises(profiles.ProfileError, match="schema_version"):
        profiles.parse_profile(data)


def test_corrupt_store_never_breaks_startup(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text("{not json", encoding="utf-8")
    loaded, errors = profiles.load_profiles(path)
    assert loaded == []
    assert errors and "profiles:" in errors[0]
    with pytest.raises(ValueError):
        profiles.load_profiles(path, strict=True)


def test_atomic_roundtrip_and_corrupt_entry_skipping(tmp_path):
    path = tmp_path / "profiles.json"
    items = [builtin("builtin-web-photo"), builtin("builtin-thumbnails")]
    profiles.save_profiles(path, items)
    loaded, errors = profiles.load_profiles(path)
    assert [item.to_dict() for item in loaded] == [item.to_dict() for item in items]
    assert errors == []
    assert not list(tmp_path.glob("*.tmp"))


def test_import_renames_id_and_name_collisions():
    existing = [builtin("builtin-web-photo")]
    imported = [builtin("builtin-web-photo")]
    combined = profiles.import_into(existing, imported)
    assert len(combined) == 2
    assert combined[1].id != combined[0].id
    assert combined[1].name != combined[0].name
    assert combined[1].quality == combined[0].quality


def test_unique_id_is_safe_and_stable_for_ascii_names():
    existing = [builtin("builtin-web-photo")]
    assert profiles.unique_id(existing, "My Profile") == "my-profile"
    assert profiles.PROFILE_ID_RE.fullmatch(
        profiles.unique_id(existing, "Мой профиль"))


def test_collection_file_with_two_profiles_is_rejected_by_cli_loader(tmp_path):
    path = tmp_path / "two.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "profiles": [builtin("builtin-web-photo").to_dict(),
                     builtin("builtin-thumbnails").to_dict()],
    }, ensure_ascii=False), encoding="utf-8")
    parser = cli.build_parser()
    args = parser.parse_args(["--profile-file", str(path)])
    with pytest.raises(SystemExit, match="exactly one"):
        cli._load_cli_profile(args)


def test_old_cli_defaults_are_unchanged():
    parser = cli.build_parser()
    args = parser.parse_args([])
    opts = cli._make_options(args, Path("in"),
                             Path("out"), None, set())
    assert (opts.max_w, opts.max_h) == (1024, 1024)
    assert opts.mode == "contain"
    assert opts.quality == 80
    assert opts.subsampling == 2
    assert opts.sizing == "resolution"
    assert opts.allow_upscale is False
    assert opts.keep_exif is True
    assert opts.keep_icc is True


def test_cli_profile_sets_geometry_and_metadata():
    parser = cli.build_parser()
    argv = ["--profile", "builtin-thumbnails"]
    args = parser.parse_args(argv)
    profile = cli._load_cli_profile(args)
    opts = cli._make_options(args, Path("in"),
                             Path("out"), profile,
                             cli._provided_dests(argv, parser))
    assert (opts.max_w, opts.max_h) == (320, 320)
    assert opts.mode == "cover"
    assert opts.quality == 80
    assert opts.subsampling == 2


def test_explicit_cli_values_override_profile():
    parser = cli.build_parser()
    argv = ["--profile", "builtin-thumbnails", "--size", "custom",
            "--width", "500", "--height", "400", "--mode", "contain",
            "--jpeg-quality", "70", "--subsampling", "444"]
    args = parser.parse_args(argv)
    profile = cli._load_cli_profile(args)
    opts = cli._make_options(args, Path("in"),
                             Path("out"), profile,
                             cli._provided_dests(argv, parser))
    assert (opts.max_w, opts.max_h) == (500, 400)
    assert opts.mode == "contain"
    assert opts.quality == 70
    assert opts.subsampling == 0


def test_cli_profile_file_roundtrip(tmp_path):
    path = tmp_path / "web.json"
    profiles.export_profile(path, builtin("builtin-web-photo"))
    parser = cli.build_parser()
    args = parser.parse_args(["--profile-file", str(path)])
    profile = cli._load_cli_profile(args)
    assert profile is not None
    assert profile.name == "Фото для сайта"
    assert profile.quality == 85


def test_parse_colour_shared_helper():
    assert profiles.parse_colour("#ff8000") == (255, 128, 0)
    with pytest.raises(profiles.ProfileError):
        profiles.parse_colour("orange")


def test_profile_serialisation_is_a_deep_copy():
    original = builtin("builtin-web-photo")
    data = original.to_dict()
    data["jpeg"]["quality"] = 1
    assert original.quality == 85
    assert copy.deepcopy(original.to_dict())["jpeg"]["quality"] == 85

