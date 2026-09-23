import json

import pytest

from haltere.liftoff.flight_card import render_card, write_card


def test_terminal_impact_is_failure_even_when_csv_contact_list_is_empty():
    score = {'flight.csv': [{'collisions': [], 'terminal_impact': {'unexplained_mps2': 63.4}}]}
    card = render_card({'track': 'Development course'}, score)
    assert 'Failed: impact recorded' in card
    assert 'CSV contact estimates: 0' in card
    assert '63.4' in card


def test_missing_result_or_hud_evidence_never_becomes_a_finish():
    assert 'Preflight: result pending' in render_card({})
    assert 'No confirmed game finish' in render_card({}, {'flight.csv': [{}]})
    with pytest.raises(ValueError, match='finish evidence'):
        render_card({}, {}, {'finish_confirmed': True})
    card = render_card({}, {'flight.csv': [{'collisions': [1]}]},
                       {'finish_confirmed': True, 'finish_evidence': ['finish.png']})
    assert 'contact criterion failed' in card


def test_card_requires_saved_evidence_and_never_overwrites_previous_card(tmp_path):
    manifest, review = tmp_path/'manifest.json', tmp_path/'review.json'
    manifest.write_text('{}')
    missing = tmp_path/'missing.png'
    review.write_text(json.dumps({'finish_confirmed': True, 'finish_evidence': [str(missing)]}))
    out = tmp_path/'card.md'
    with pytest.raises(ValueError, match='Missing saved'):
        write_card(manifest, out, review_path=review)
    assert not out.exists()
    write_card(manifest, out)
    with pytest.raises(FileExistsError):
        write_card(manifest, out)


def test_oracle_finish_is_never_presented_as_an_autonomous_evaluation():
    card=render_card({'runtime_route_oracle':True}, {},
                     {'finish_confirmed':True,'finish_evidence':['finish.png']})
    assert 'Oracle collection only; ineligible for autonomous evaluation.' in card
    assert 'Game finish confirmed.' in card
