"""Background worker: takes ingest and summarise jobs off the Redis stream."""
import logging
import socket
import time

import redis

import context
import persistence as db
import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("worker")

HANDLERS = {"ingest": pipeline.ingest, "summarize": pipeline.summarize}


def main():
    db.init()
    r = context.client()
    while True:
        try:
            r.ping()
            break
        except redis.ConnectionError:
            log.info("Waiting for Redis...")
            time.sleep(1)
    context.ensure_group(r)
    context.ensure_index(r)
    name = socket.gethostname()
    log.info("Worker %s ready", name)

    while True:
        resp = r.xreadgroup(context.GROUP, name, {context.STREAM: ">"}, count=1, block=2000)
        for _stream, entries in resp or []:
            for msg_id, job in entries:
                log.info("Job %s: %s", msg_id, job)
                started = time.time()
                try:
                    HANDLERS[job["type"]](r, job["doc_id"])
                    log.info("Done in %.1fs", time.time() - started)
                except Exception:
                    log.exception("Job failed")  # status is already recorded in SQLite
                finally:
                    r.xack(context.STREAM, context.GROUP, msg_id)


if __name__ == "__main__":
    main()
