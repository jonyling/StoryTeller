from pipeline.companion import (
    ask_companion,
    canon_from_pages,
    new_session,
    retrieve_passages,
)


def _pages():
    return [
        {
            "page": 1,
            "sentences": [
                {"text": "Ember lived in a quiet valley.", "speaker": "narrator", "emotion": "calm"},
                {"text": "A storm arrived suddenly.", "speaker": "narrator", "emotion": "angry"},
                {"text": "Ember found golden light.", "speaker": "narrator", "emotion": "excited"},
            ],
        }
    ]


def test_canon_and_strict_retrieval_no_spoilers():
    session = new_session(_pages(), language="English")
    session.advance_to(0)
    passages = retrieve_passages(session, "Where did Ember live?")
    assert all(p["index"] <= 0 for p in passages)
    assert any("valley" in p["text"] for p in passages)


def test_ask_companion_uses_reasoner_and_memory():
    class FakeReasoner:
        def answer(self, session, question, passages):
            assert passages
            assert session.turns[-1].role == "user"
            return "Ember lived in a quiet valley."

    session = new_session(_pages())
    session.advance_to(1)
    answer = ask_companion(session, "Where did Ember live?", FakeReasoner())
    assert "valley" in answer
    assert len(session.turns) == 2
    assert session.turns[0].role == "user"
    assert session.turns[1].role == "assistant"


def test_canon_from_pages_indexes():
    canon = canon_from_pages(_pages())
    assert [c["index"] for c in canon] == [0, 1, 2]
