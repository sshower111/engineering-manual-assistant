from fakes import FakeAnthropicClient, chunk

from manual_assistant.llm import ClaudeAnswerGenerator, LLMAnswer, build_user_message

CHUNKS = [chunk("hash-a", 0.8, "Fixed constraint text."), chunk("hash-b", 0.6, "Pad text.")]


def generator(client):
    return ClaudeAnswerGenerator("claude-sonnet-5-5", max_tokens=1000, effort="low", client=client)


def test_user_message_numbers_excerpts_and_includes_question():
    message = build_user_message("How?", CHUNKS)
    assert '<excerpt id="C1" source="FEM ConstraintFixed > Usage">' in message
    assert '<excerpt id="C2"' in message
    assert "Fixed constraint text." in message
    assert message.endswith("Question: How?")


def test_request_uses_structured_output_and_configured_model():
    client = FakeAnthropicClient(LLMAnswer(answerable=True, answer="A", cited_ids=["C1"]))
    generator(client).generate("How?", CHUNKS)

    request = client.requests[0]
    assert request["model"] == "claude-sonnet-5-5"
    assert request["output_format"] is LLMAnswer
    assert request["output_config"] == {"effort": "low"}
    assert "Use only the documentation excerpts" in request["system"]


def test_cited_ids_map_back_to_chunks_and_unknown_ids_are_dropped():
    parsed = LLMAnswer(answerable=True, answer=" Answer. ", cited_ids=["C2", "C9", "C2"])
    result = generator(FakeAnthropicClient(parsed)).generate("How?", CHUNKS)

    assert result.answerable
    assert result.answer == "Answer."
    assert [c.id for c in result.cited_chunks] == ["hash-b"]
    assert (result.input_tokens, result.output_tokens) == (100, 20)


def test_unanswerable_or_empty_answer_is_a_refusal():
    for parsed in (
        LLMAnswer(answerable=False, answer="", cited_ids=[]),
        LLMAnswer(answerable=True, answer="   ", cited_ids=["C1"]),
    ):
        result = generator(FakeAnthropicClient(parsed)).generate("How?", CHUNKS)
        assert not result.answerable
        assert result.cited_chunks == []


def test_api_refusal_stop_reason_is_a_refusal():
    client = FakeAnthropicClient(parsed=None, stop_reason="refusal")
    assert not generator(client).generate("How?", CHUNKS).answerable
