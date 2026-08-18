from collections import Counter
from pathlib import Path
from unittest.mock import patch

from learning import DeliveryReceipt, LearningService, LearningSettings, Producer
from learning.normalized_event import normalize_telegram_event

from test_r6_service_decomposition import Provider, message


def test_r6_200_event_component_trace(tmp_path):
    """Exercise the production facade boundary with the requested R6 mix.

    Algorithmic behavior remains covered by R0-R5 integration suites; this
    diagnostic protects the new ownership/delegation graph from fan-out.
    """
    service = LearningService(
        LearningSettings(data_dir=Path(tmp_path), openai_chat_id=-1),
        llm_provider=Provider(),
    )
    categories = (
        ["direct"] * 60
        + ["social_local"] * 30
        + ["pending"] * 20
        + ["ordinary"] * 25
        + ["media"] * 20
        + ["autonomous"] * 15
        + ["provider_failure"] * 10
        + ["delivery_failure"] * 10
        + ["no_response"] * 10
    )
    trace = Counter()
    current_event_for_plan = None

    def plan_for(event, producer=Producer.LOCAL):
        return service.prepare_text_response(
            event, "prepared", producer=producer, purpose="r6_smoke"
        )

    def foreground_result(event, *args, **kwargs):
        trace["foreground"] += 1
        return plan_for(event)

    def media_result(event, *args, **kwargs):
        trace["media"] += 1
        return plan_for(event, Producer.MEDIA)

    def autonomous_result(*args, **kwargs):
        trace["autonomous"] += 1
        return plan_for(current_event_for_plan)

    with (
        patch.object(service.foreground, "prepare_direct_reply", side_effect=foreground_result),
        patch.object(service.foreground, "maybe_reply", side_effect=foreground_result),
        patch.object(service.foreground, "prepare_pending_continuation", side_effect=foreground_result),
        patch.object(service.foreground, "prepare_reply", side_effect=foreground_result),
        patch.object(service.media_coordinator, "maybe_command_meme", side_effect=media_result),
        patch.object(service.autonomous, "prepare_autonomous", side_effect=autonomous_result),
        patch.object(service.generation, "generate_llm", return_value=None) as failed_provider,
    ):
        for index, category in enumerate(categories, start=1):
            incoming = message(index, chat_id=-(index % 5 + 1))
            event = normalize_telegram_event(incoming)
            current_event_for_plan = event
            trace["events"] += 1
            assert event.event_id
            plan = None
            if category == "direct":
                plan = service.prepare_direct_reply(event, explicit_address=True)
            elif category == "social_local":
                plan = service.maybe_reply(event)
            elif category == "pending":
                plan = service.prepare_pending_continuation(event)
            elif category == "ordinary":
                plan = service.prepare_reply(event)
            elif category == "media":
                plan = service.maybe_command_meme(event)
            elif category == "autonomous":
                plan = service.prepare_autonomous(event.chat_id, event.timestamp)
            elif category == "provider_failure":
                assert service.generate_llm(event.chat_id, "context") is None
                trace["provider_failure"] += 1
            elif category == "delivery_failure":
                plan = service.prepare_text_response(
                    event, "prepared", producer=Producer.LOCAL
                )
                trace["delivery_failure"] += 1
            else:
                trace["no_response"] += 1

            if plan is not None:
                trace["plans"] += 1
                assert plan.event_id == event.event_id
                failed = category == "delivery_failure"
                receipt = DeliveryReceipt(
                    event_id=event.event_id,
                    success=not failed,
                    delivery_type=plan.delivery_type,
                    telegram_message_id=None if failed else 10_000 + index,
                    error_category="telegram_timeout" if failed else None,
                )
                trace["delivery_attempts"] += 1
                committed = service.finalize_response(plan, receipt)
                trace["aborts" if failed else "commits"] += 1
                assert committed is True

    assert trace == Counter(
        events=200,
        foreground=135,
        media=20,
        autonomous=15,
        provider_failure=10,
        delivery_failure=10,
        no_response=10,
        plans=180,
        delivery_attempts=180,
        commits=170,
        aborts=10,
    )
    assert failed_provider.call_count == 10
