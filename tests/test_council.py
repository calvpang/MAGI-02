"""Tests for the MAGI council."""



def test_agent_personality_import():
    """Test that agent personalities can be imported."""
    from magi.agents.council import BALTHASAR, CASPER, MELCHIOR

    assert MELCHIOR.name == "MELCHIOR-1"
    assert MELCHIOR.designation == "Scientist"

    assert BALTHASAR.name == "BALTHASAR-2"
    assert BALTHASAR.designation == "Mother"

    assert CASPER.name == "CASPER-3"
    assert CASPER.designation == "Woman"


def test_agent_personality_has_system_prompt():
    """Test that each personality has a non-empty system prompt."""
    from magi.agents.council import BALTHASAR, CASPER, MELCHIOR

    assert len(MELCHIOR.system_prompt) > 100
    assert len(BALTHASAR.system_prompt) > 100
    assert len(CASPER.system_prompt) > 100


def test_magi_council_import():
    """Test that MAGICouncil can be imported."""
    from magi.agents.council import MAGICouncil

    assert MAGICouncil is not None
