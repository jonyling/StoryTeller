from pipeline.prosody import map_dsp_params, strip_delivery_tags
from pipeline.story_gen import StorySentence
from pipeline.theatre import RuleBasedTheatreAdapter, theatre_lines_to_script_doc


def test_strip_delivery_tags():
    assert strip_delivery_tags("Hello [whispers] there.") == "Hello there."


def test_map_dsp_params_neutral_is_identity():
    assert map_dsp_params(3, 3, 3) == (0.0, 1.0, 0.0)


def test_rule_based_theatre_adds_stage_and_tags():
    adapter = RuleBasedTheatreAdapter()
    lines = adapter.adapt([
        StorySentence(text="Once [gasps] upon a time.", speaker="narrator", emotion="calm"),
        StorySentence(text="Run!", speaker="Ember", emotion="excited"),
    ])
    assert lines[0].speak_text == "Once upon a time."
    assert lines[0].pitch == 3  # narrator: no pitch shift
    assert lines[0].rate == 2   # calm narrator: slightly slower only
    assert "Ember" in lines[1].stage_direction
    doc = theatre_lines_to_script_doc(lines)
    assert doc["schema_version"] == "theatre-v1"
    assert len(doc["acts"][0]["scenes"][0]["lines"]) == 2
