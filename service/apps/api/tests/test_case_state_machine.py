import pytest

from drynaria_api.models import CaseStatus, CaseStateError, apply_case_transition, can_transition_case


@pytest.mark.parametrize(
    ("current", "next_status"),
    [
        (CaseStatus.Draft, CaseStatus.Ready),
        (CaseStatus.Draft, CaseStatus.NeedsRetake),
        (CaseStatus.Ready, CaseStatus.Queued),
        (CaseStatus.Ready, CaseStatus.NeedsRetake),
        (CaseStatus.Ready, CaseStatus.Draft),
        (CaseStatus.NeedsRetake, CaseStatus.Ready),
        (CaseStatus.Queued, CaseStatus.Analyzing),
        (CaseStatus.Analyzing, CaseStatus.Completed),
        (CaseStatus.Analyzing, CaseStatus.NeedsRetake),
        (CaseStatus.Analyzing, CaseStatus.Inconclusive),
        (CaseStatus.Analyzing, CaseStatus.Failed),
    ],
)
def test_allowed_case_transitions(current, next_status):
    assert can_transition_case(current, next_status)


@pytest.mark.parametrize(
    ("current", "next_status"),
    [
        (CaseStatus.Completed, CaseStatus.Analyzing),
        (CaseStatus.Completed, CaseStatus.Queued),
        (CaseStatus.Failed, CaseStatus.Completed),
        (CaseStatus.Inconclusive, CaseStatus.Ready),
        (CaseStatus.Draft, CaseStatus.Analyzing),
        (CaseStatus.Ready, CaseStatus.Completed),
        (CaseStatus.Analyzing, CaseStatus.Queued),
    ],
)
def test_denied_case_transitions(current, next_status):
    assert not can_transition_case(current, next_status)


def test_apply_transition_raises_on_invalid_move():
    from types import SimpleNamespace

    case = SimpleNamespace(status=CaseStatus.Completed)
    with pytest.raises(CaseStateError):
        apply_case_transition(case, CaseStatus.Analyzing)
