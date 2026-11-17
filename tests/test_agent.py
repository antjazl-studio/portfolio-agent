"""Deterministic tests: no network, no model, no key. They run on every PR.

LLM behaviour is measured by promptfoo (probabilistic). These tests pin down the
guarantees we put in CODE, so a refactor can't silently remove one.
"""
import json
from types import SimpleNamespace

import pytest

import agent


# --- A scripted fake of the OpenAI client --------------------------------------
def tool_call(name, arguments="{}", call_id="call_1"):
    call = SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))
    call.model_dump = lambda: {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    return call


def reply(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


@pytest.fixture
def fake_model(monkeypatch):
    """Replaces the real API call; returns the list of kwargs each call received."""
    calls = []

    def install(*responses):
        queue = list(responses)

        def create(**kwargs):
            calls.append(kwargs)
            return queue.pop(0)

        monkeypatch.setattr(agent.client.chat.completions, "create", create)
        return calls

    return install


def ask(question="hi"):
    return agent.run_agent([
        {"role": "system", "content": agent.SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ])


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


# --- The loop ----------------------------------------------------------------------
def test_first_step_forces_a_tool_call_then_lets_the_model_decide(fake_model):
    calls = fake_model(
        reply(tool_calls=[tool_call("get_contact_info")]),
        reply(content="His email is antjazl@gmail.com."),
    )
    ask("What's his email?")
    assert calls[0]["tool_choice"] == "required"
    assert calls[1]["tool_choice"] == "auto"


def test_text_written_alongside_a_tool_call_is_not_lost(fake_model):
    fake_model(
        reply(content="Yes, he's Azure certified.", tool_calls=[tool_call("get_contact_info")]),
        reply(content="His CV is at https://luisjacho.dev/Luis_Jacho_Resume.pdf"),
    )
    answer = ask("Is he certified?")
    assert "Azure certified" in answer
    assert "Luis_Jacho_Resume.pdf" in answer


def test_invented_email_never_reaches_the_visitor(fake_model):
    fake_model(
        reply(tool_calls=[tool_call("get_contact_info")]),
        reply(content="Sure: luiz@jacho.dev"),
    )
    assert ask("What's his email?") == agent.SAFE_ANSWER


def test_bad_tool_arguments_go_back_to_the_model_instead_of_crashing(fake_model):
    calls = fake_model(
        reply(tool_calls=[tool_call("search_profile", arguments="{not json")]),
        reply(content="Let me answer anyway."),
    )
    ask()
    tool_messages = [m for m in calls[1]["messages"] if m["role"] == "tool"]
    assert tool_messages[0]["content"].startswith("Tool error")


def test_provider_returning_no_choices_gives_the_fallback_not_a_crash(fake_model):
    empty = SimpleNamespace(choices=None, error={"message": "upstream timeout"})
    calls = fake_model(empty, empty)
    assert ask() == agent.FALLBACK_ANSWER
    assert len(calls) == 2  # retried once, then gave up


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
