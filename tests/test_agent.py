"""Deterministic tests: no network, no model, no key. They run on every PR.

LLM behaviour is measured by promptfoo (probabilistic). These tests pin down the
guarantees we put in CODE, so a refactor can't silently remove one.
"""
import json
from types import SimpleNamespace

import pytest

import agent


# --- A scripted fake of the streaming OpenAI client ----------------------------
def tool_call(name, arguments="{}", call_id="call_1"):
    return {"name": name, "arguments": arguments, "id": call_id}


def reply(content=None, tool_calls=None, pieces=3):
    """One model turn as a list of stream chunks, the way the API sends it:
    text in small deltas, tool calls in fragments, usage in a final empty chunk."""
    chunks = []

    def chunk(content=None, tool_calls=None):
        delta = SimpleNamespace(content=content, tool_calls=tool_calls)
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)

    if content:
        size = max(1, len(content) // pieces)
        for i in range(0, len(content), size):
            chunks.append(chunk(content=content[i:i + size]))
    for index, call in enumerate(tool_calls or []):
        half = len(call["arguments"]) // 2
        first = SimpleNamespace(index=index, id=call["id"],
                                function=SimpleNamespace(name=call["name"], arguments=call["arguments"][:half]))
        rest = SimpleNamespace(index=index, id=None,
                               function=SimpleNamespace(name=None, arguments=call["arguments"][half:]))
        chunks += [chunk(tool_calls=[first]), chunk(tool_calls=[rest])]
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    chunks.append(SimpleNamespace(choices=[], usage=usage))
    return chunks


def broken_stream():
    return [SimpleNamespace(choices=[], usage=None, error={"message": "upstream timeout"})]


@pytest.fixture
def fake_model(monkeypatch):
    """Replaces the real API call; returns the list of kwargs each call received."""
    calls = []

    def install(*turns):
        queue = list(turns)

        def create(**kwargs):
            calls.append(kwargs)
            return iter(queue.pop(0))

        monkeypatch.setattr(agent.client.chat.completions, "create", create)
        return calls

    return install


def conversation(question="hi"):
    return [
        {"role": "system", "content": agent.SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def ask(question="hi"):
    return agent.run_agent(conversation(question))


def events(question="hi"):
    return list(agent.stream_agent(conversation(question)))


# --- Output guardrail ------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "Email him at antjazl@gmail.com.",
    "LinkedIn: https://linkedin.com/in/luis-jacho-zambrano-843558139",
    "[CV](https://luisjacho.dev/Luis_Jacho_Resume.pdf)",
    "See luisjacho.dev for more.",
    "He works with Node.js, ASP.NET and Angular.",
])
def test_guardrail_allows_real_contact_and_tech_words(text):
    assert agent.guard_output(text) == text


@pytest.mark.parametrize("text", [
    "Email him at luisjacho.zambrano@gmail.com",  # real invention from an eval run
    "Write to luis@jacho.dev",                    # real invention from an eval run
    "Profile: https://linkedin.com/in/someone-else",
    "Download it at https://evil.example.com/cv.pdf",
])
def test_guardrail_blocks_unknown_emails_and_links(text):
    assert agent.guard_output(text) == agent.SAFE_ANSWER


# --- Streaming + guardrail ----------------------------------------------------------
def test_text_is_streamed_in_several_events(fake_model):
    fake_model(
        reply(tool_calls=[tool_call("search_profile", '{"query": "azure"}')]),
        reply(content="He built Azure Functions. He set up autoscaling. He is certified."),
    )
    text_events = [e for e in events() if e["type"] == "text"]
    assert len(text_events) > 1
    assert "".join(e["text"] for e in text_events) == (
        "He built Azure Functions. He set up autoscaling. He is certified."
    )


def test_invented_email_is_never_shown_even_partially(fake_model):
    fake_model(
        reply(tool_calls=[tool_call("get_contact_info")]),
        reply(content="Here is his contact. Email: luiz@jacho.dev for anything.", pieces=12),
    )
    stream = events()
    shown = "".join(e["text"] for e in stream if e["type"] == "text")
    assert "jacho.dev" not in shown and "luiz" not in shown
    assert {"type": "replace", "text": agent.SAFE_ANSWER} in stream
    assert stream[-1] == {"type": "done"}


def test_invented_email_never_reaches_the_visitor(fake_model):
    fake_model(
        reply(tool_calls=[tool_call("get_contact_info")]),
        reply(content="Sure: luiz@jacho.dev"),
    )
    assert ask("What's his email?") == agent.SAFE_ANSWER


def test_tool_use_is_announced_to_the_widget(fake_model):
    fake_model(
        reply(tool_calls=[tool_call("search_profile", '{"query": "x"}')]),
        reply(content="Done."),
    )
    assert {"type": "tool", "name": "search_profile"} in events()


# --- The loop ----------------------------------------------------------------------
def test_first_step_forces_a_tool_call_then_lets_the_model_decide(fake_model):
    calls = fake_model(
        reply(tool_calls=[tool_call("get_contact_info")]),
        reply(content="His email is antjazl@gmail.com."),
    )
    ask("What's his email?")
    assert calls[0]["tool_choice"] == "required"
    assert calls[1]["tool_choice"] == "auto"
    assert all(c["stream"] for c in calls)


def test_text_written_alongside_a_tool_call_is_not_lost(fake_model):
    fake_model(
        reply(content="Yes, he's Azure certified.", tool_calls=[tool_call("get_contact_info")]),
        reply(content="His CV is at https://luisjacho.dev/Luis_Jacho_Resume.pdf"),
    )
    answer = ask("Is he certified?")
    assert "Azure certified" in answer
    assert "Luis_Jacho_Resume.pdf" in answer


def test_tool_call_fragments_are_reassembled(fake_model):
    calls = fake_model(
        reply(tool_calls=[tool_call("search_profile", '{"query": "kubernetes"}')]),
        reply(content="Not in his profile."),
    )
    ask()
    tool_message = [m for m in calls[1]["messages"] if m["role"] == "tool"][0]
    assert "Tool error" not in tool_message["content"]  # JSON arrived whole


def test_bad_tool_arguments_go_back_to_the_model_instead_of_crashing(fake_model):
    calls = fake_model(
        reply(tool_calls=[tool_call("search_profile", arguments="{not json")]),
        reply(content="Let me answer anyway."),
    )
    ask()
    tool_messages = [m for m in calls[1]["messages"] if m["role"] == "tool"]
    assert tool_messages[0]["content"].startswith("Tool error")


def test_provider_error_is_retried_once_then_gives_the_fallback(fake_model):
    calls = fake_model(broken_stream(), broken_stream())
    assert ask() == agent.FALLBACK_ANSWER
    assert len(calls) == 2  # retried once, then gave up


def test_provider_error_then_success_is_invisible_to_the_visitor(fake_model):
    fake_model(
        broken_stream(),
        reply(tool_calls=[tool_call("get_contact_info")]),
        reply(content="His email is antjazl@gmail.com."),
    )
    assert ask() == "His email is antjazl@gmail.com."


def test_empty_stream_gives_the_fallback_not_a_crash(fake_model):
    empty = [SimpleNamespace(choices=[], usage=None)]
    fake_model(empty, empty)
    assert ask() == agent.FALLBACK_ANSWER


def test_step_limit_returns_the_fallback(fake_model):
    endless = [reply(tool_calls=[tool_call("get_contact_info")]) for _ in range(agent.MAX_STEPS)]
    fake_model(*endless)
    assert ask() == agent.FALLBACK_ANSWER


def test_tool_results_follow_the_assistant_message_that_requested_them(fake_model):
    calls = fake_model(
        reply(tool_calls=[tool_call("get_contact_info", call_id="abc")]),
        reply(content="Done."),
    )
    ask()
    history = calls[1]["messages"]  # same list the loop keeps appending to
    i = next(i for i, m in enumerate(history) if m["role"] == "tool")
    assert history[i - 1]["role"] == "assistant"
    assert history[i - 1]["tool_calls"][0]["id"] == "abc"
    assert history[i]["tool_call_id"] == "abc"
    assert json.dumps(history)  # history stays JSON-serializable
