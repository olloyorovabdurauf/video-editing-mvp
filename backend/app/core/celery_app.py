"""
Celery app factory.

We split work across queues by resource profile so we can scale each kind
of worker independently. See docker-compose for separate worker services.
"""
from celery import Celery
from celery.signals import worker_process_init

from app.config import get_settings

settings = get_settings()


# Init Sentry inside each worker process (not the parent), because Celery
# pre-forks workers and signal-based integrations only attach correctly
# after the fork.
@worker_process_init.connect
def _init_worker(**_) -> None:
    from app.core.observability import init_sentry
    from app.db.session import init_engine
    init_sentry()
    init_engine()                      # no-op unless DATABASE_URL is set

celery_app = Celery(
    "video_saas",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.video_tasks", "app.tasks.creative_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Resilience: long-running jobs survive worker restarts and don't get
    # silently re-delivered to a second worker mid-run.
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    # Routing: each task is registered with a queue, see decorators.
    task_default_queue="io",
    task_queues={
        "io":     {"exchange": "io",     "routing_key": "io"},
        "ai":     {"exchange": "ai",     "routing_key": "ai"},
        "ffmpeg": {"exchange": "ffmpeg", "routing_key": "ffmpeg"},
        "gpu":    {"exchange": "gpu",    "routing_key": "gpu"},
    },
)
