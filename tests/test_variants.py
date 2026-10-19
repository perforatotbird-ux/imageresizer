"""Contract tests for explicit per-source output variants.

These cover the small parsing/validation layer only; end-to-end pipeline
behaviour is exercised in ``test_variants_pipeline.py``.
"""

import pytest

from batchresize.variants import (OutputVariant, VariantError, parse_variant_spec,
                                  validate_id)


class TestParseVariantSpec:
    def test_resolution_form(self):
        v = parse_variant_spec("thumb:320x320:cover:80:420")
        assert v.id == "thumb"
        assert (v.max_w, v.max_h) == (320, 320)
        assert v.mode == "cover"
        assert v.quality == 80
        assert v.subsampling == "420"

    def test_defaults_applied(self):
        v = parse_variant_spec("web:1280")
        assert (v.max_w, v.max_h) == (1280, 1280)
        assert v.mode == "contain"
        assert v.quality == 80
        assert v.subsampling == "auto"

    def test_longest_form(self):
        v = parse_variant_spec("small:longest:320")
        assert v.sizing == "longest"
        assert v.sizing_value == 320

    def test_shortest_form(self):
        v = parse_variant_spec("small:shortest:200")
        assert v.sizing == "shortest"
        assert v.sizing_value == 200

    def test_percent_form(self):
        v = parse_variant_spec("half:percent:50:cover")
        assert v.sizing == "percent"
        assert v.sizing_value == 50
        assert v.mode == "cover"

    def test_megapixels_form(self):
        v = parse_variant_spec("mp:megapixels:2")
        assert v.sizing == "megapixels"
        assert v.sizing_value == 2

    def test_subsampling_and_quality_shift_for_sizing(self):
        v = parse_variant_spec("s:longest:320:cover:70:444")
        assert v.mode == "cover"
        assert v.quality == 70
        assert v.subsampling == "444"

    def test_id_is_validated(self):
        with pytest.raises(VariantError, match="variant.id"):
            parse_variant_spec("Bad ID!:320")

    def test_too_few_parts_rejected(self):
        with pytest.raises(VariantError):
            parse_variant_spec("onlyid")

    def test_bad_size_rejected(self):
        with pytest.raises(VariantError):
            parse_variant_spec("id:notasize")

    def test_bad_quality_rejected(self):
        with pytest.raises(VariantError):
            parse_variant_spec("id:320x320:x:cover:high")


class TestOutputVariantValidation:
    def test_quality_out_of_range_rejected(self):
        with pytest.raises(VariantError):
            OutputVariant(id="a", quality=0)

    def test_bad_mode_rejected(self):
        with pytest.raises(VariantError):
            OutputVariant(id="a", mode="pad")

    def test_bad_subsampling_rejected(self):
        with pytest.raises(VariantError):
            OutputVariant(id="a", subsampling="999")

    def test_bad_bg_color_rejected(self):
        with pytest.raises(VariantError):
            OutputVariant(id="a", bg_color="white")

    def test_unknown_sizing_rejected(self):
        with pytest.raises(VariantError):
            OutputVariant(id="a", sizing="diagonal", sizing_value=1)

    def test_resolution_requires_positive_box(self):
        with pytest.raises(VariantError):
            OutputVariant(id="a", max_w=0)

    def test_name_defaults_to_id(self):
        assert OutputVariant(id="thumb").name == "thumb"


class TestProfileRoundtrip:
    def test_to_from_profile_dict_is_stable(self):
        v = OutputVariant(id="web", name="Web", max_w=1280, max_h=1280,
                          mode="contain", quality=85, subsampling="444",
                          bg_color="#EEEEEE")
        data = v.to_profile_dict()
        assert data["id"] == "web"
        assert data["name"] == "Web"
        assert data["size"]["mode"] == "resolution"
        assert data["fit"]["mode"] == "contain"
        assert data["jpeg"]["quality"] == 85
        assert data["jpeg"]["subsampling"] == "444"
        assert data["jpeg"]["bg_color"] == "#EEEEEE"
        restored = OutputVariant.from_profile_dict(data)
        assert restored.to_profile_dict() == data

    def test_non_resolution_size_preserved(self):
        v = OutputVariant(id="small", sizing="longest", sizing_value=320)
        data = v.to_profile_dict()
        restored = OutputVariant.from_profile_dict(data)
        assert restored.sizing == "longest"
        assert restored.sizing_value == 320

    def test_invalid_profile_dict_rejected(self):
        with pytest.raises(VariantError):
            OutputVariant.from_profile_dict({"id": "a", "mode": "pad"})

    def test_validate_id(self):
        assert validate_id("ok-1.id") == "ok-1.id"
        with pytest.raises(VariantError):
            validate_id("Bad!")