from Defenses.ImageDefenses.Defend import Defender


def test_defender_uses_saved_attachment_mode_by_default():
    trigger = {'how_to_attach': 'replace'}

    assert Defender._resolve_trigger_attachment(trigger, None) == 'replace'


def test_defender_allows_explicit_attachment_override():
    trigger = {'how_to_attach': 'replace'}

    assert Defender._resolve_trigger_attachment(trigger, 'blend') == 'blend'


def test_defender_defaults_legacy_trigger_attachment_to_blend():
    assert Defender._resolve_trigger_attachment({}, None) == 'blend'


def test_defender_uses_saved_source_filter_by_default():
    trigger = {'source_filter': 'good'}

    assert Defender._resolve_source_filter(trigger, None) == 'good'
    assert Defender._resolve_source_filter(trigger, 'bad') == 'bad'
    assert Defender._resolve_source_filter({}, None) == 'bad'
