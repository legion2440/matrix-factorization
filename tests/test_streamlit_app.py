from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_profile_shortcut_synchronizes_keyed_manual_input():
    profiles = json.loads(
        (ROOT / "reports" / "evaluated_users.json").read_text(encoding="utf-8")
    )
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)

    at.run()
    assert not at.exception
    assert at.text_input[0].value == str(profiles[0]["user_id"])

    at.selectbox[0].select(profiles[1]).run()
    assert not at.exception
    assert at.text_input[0].value == str(profiles[1]["user_id"])

    manual_value = str(profiles[2]["user_id"])
    at.text_input[0].input(manual_value).run()
    assert not at.exception
    assert at.text_input[0].value == manual_value

    at.text_input[0].input("not-a-number").run()
    assert not at.exception
    assert any("Invalid user ID input" in error.value for error in at.error)

    at.text_input[0].input("999999999").run()
    assert not at.exception
    assert any("Unknown user ID" in error.value for error in at.error)
