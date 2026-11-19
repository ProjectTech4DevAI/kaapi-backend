#!/usr/bin/env python3
"""
Post a daily digest of Sentry errors for one project to a Discord channel.

Fetches every unresolved issue seen in the last N hours, then posts one summary
card plus one card per issue (sorted by Sentry priority, then volume).

Usage:
    export SENTRY_AUTH_TOKEN=...      # scopes: event:read, project:read, org:read
    python scripts/python/sentry_daily_digest.py --project kaapi-staging          # dry run
    DISCORD_WEBHOOK_URL=... DISCORD_ROLE_ID=... \\
        python scripts/python/sentry_daily_digest.py --project kaapi-staging --send
"""

import argparse
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Sentry
SENTRY_BASE_URL = os.environ.get("SENTRY_BASE_URL", "https://sentry.io")
SENTRY_ORG = os.environ.get("SENTRY_ORG", "project-tech4dev")
HTTP_TIMEOUT_SECONDS = 30
ISSUE_FETCH_LIMIT = 100

# Priority: Sentry's own value wins; the fallback only applies when Sentry has none.
PRIORITY_LABEL = {"high": "High", "medium": "Medium", "low": "Low"}
PRIORITY_RANK = {"high": 2, "medium": 1, "low": 0}
FALLBACK_HIGH_VOLUME = 100
HIGHLIGHT_SUBSTATUS = {"escalating", "regressed"}

# Discord
COLOR_RED = 0xE74C3C
COLOR_ORANGE = 0xE67E22
COLOR_YELLOW = 0xF1C40F
COLOR_GREEN = 0x2ECC71
PRIORITY_COLOR = {"high": COLOR_RED, "medium": COLOR_ORANGE, "low": COLOR_YELLOW}
MAX_ISSUE_CARDS = 9  # Discord allows 10 embeds per message; one is the summary.
TITLE_MAX_CHARS = 80
DETAIL_MAX_CHARS = 200
SOURCE_MAX_CHARS = 60

# SDK exceptions (openai, anthropic) stringify the API response into the message.
API_ERROR_CODE = re.compile(r"Error code: (\d{3})")
API_ERROR_MESSAGE = re.compile(r"""['"]message['"]:\s*(['"])(.*?)\1""")
CONNECTION_POOL = re.compile(r"HTTPS?ConnectionPool\(host='([^']+)'[^)]*\)")
TIMEOUT_SUFFIX = re.compile(r"\s*\((?:read|connect) timeout=[^)]*\)")


def sentry_get(path: str, params: dict | None = None) -> dict | list:
    query = urllib.parse.urlencode(params or {}, doseq=True)
    request = urllib.request.Request(f"{SENTRY_BASE_URL}/api/0/{path}?{query}")
    request.add_header("Authorization", f"Bearer {os.environ['SENTRY_AUTH_TOKEN']}")
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return json.load(response)


def list_issues(project_slug: str, hours: int) -> tuple[str, list[dict]]:
    """Return the project id and its unresolved issues seen in the window."""
    project = sentry_get(f"projects/{SENTRY_ORG}/{project_slug}/")
    issues = sentry_get(
        f"organizations/{SENTRY_ORG}/issues/",
        {
            "project": project["id"],
            "statsPeriod": f"{hours}h",
            "query": "is:unresolved",
            "sort": "freq",
            "limit": ISSUE_FETCH_LIMIT,
        },
    )
    return project["id"], issues


def get_top_tags(issue_id: str, hours: int) -> dict[str, str]:
    """Most common value of each tag on the issue; empty if the lookup fails."""
    try:
        tags = sentry_get(
            f"organizations/{SENTRY_ORG}/issues/{issue_id}/tags/",
            {"statsPeriod": f"{hours}h"},
        )
    except urllib.error.URLError as error:
        # Tags only add Source/Release; one failed lookup should not drop the digest.
        logger.warning(f"[get_top_tags] Skipped | issue: {issue_id} | error: {error}")
        return {}
    return {
        tag["key"]: values[0]
        for tag in tags
        if (values := [v["value"] for v in tag.get("topValues", []) if v["value"]])
    }


def priority_of(issue: dict) -> str:
    priority = issue.get("priority")
    if priority in PRIORITY_LABEL:
        return priority
    if issue.get("isUnhandled") or int(issue["count"]) >= FALLBACK_HIGH_VOLUME:
        return "high"
    return "medium"


def truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def describe(issue: dict) -> tuple[str, str]:
    """Short headline plus a readable detail line, instead of Sentry's raw title."""
    meta = issue.get("metadata") or {}
    exc_type, value = meta.get("type"), meta.get("value") or ""

    if exc_type:
        code = API_ERROR_CODE.search(value)
        message = API_ERROR_MESSAGE.search(value)
        headline = f"{exc_type} ({code.group(1)})" if code else exc_type
        detail = message.group(2) if message else value
    else:
        title = meta.get("title") or issue["title"]
        title = TIMEOUT_SUFFIX.sub("", CONNECTION_POOL.sub(r"\1", title))
        headline, _, reason = title.partition(", reason: ")
        headline = headline.removesuffix(" code: None")
        detail = reason or value

    return truncate(headline, TITLE_MAX_CHARS), truncate(detail, DETAIL_MAX_CHARS)


def build_issue_card(issue: dict, is_new: bool, hours: int) -> dict:
    headline, detail = describe(issue)
    priority = priority_of(issue)
    tags = get_top_tags(issue["id"], hours)

    labels = [PRIORITY_LABEL[priority].upper()]
    if issue.get("isUnhandled"):
        labels.append("UNHANDLED")
    if is_new:
        # Sentry's own "new" substatus lasts for days; use our window so it matches the summary.
        labels.append("NEW")
    elif issue.get("substatus") in HIGHLIGHT_SUBSTATUS:
        labels.append(issue["substatus"].upper())

    count = int(issue["count"])
    last_seen = int(datetime.fromisoformat(issue["lastSeen"]).timestamp())
    task = tags.get("celery.task_name")
    source = (
        task.rsplit(".", 1)[-1]
        if task
        else tags.get("transaction") or issue.get("culprit")
    )
    release = tags.get("release")

    # Text lines rather than inline embed fields: Discord mobile stacks inline fields
    # one per row, while plain lines read the same on phone and desktop.
    lines = [" ".join(f"`{label}`" for label in labels), ""]
    if detail:
        lines.append(f"```\n{detail}\n```")
    lines += [
        "",
        f"- **Last {hours}h:** {count} {'time' if count == 1 else 'times'}",
        f"- **All time:** {issue.get('lifetime', {}).get('count', '?')}",
        f"- **Last seen:** <t:{last_seen}:R>",  # renders in each viewer's timezone
    ]
    context = []
    if source:
        context.append(f"- **Source:** `{truncate(source, SOURCE_MAX_CHARS)}`")
    if release:
        context.append(f"- **Release:** `{release}`")
    if context:
        lines += [""] + context

    assignee = (issue.get("assignedTo") or {}).get("name", "Unassigned")
    return {
        "title": headline,
        "url": issue["permalink"],
        "color": PRIORITY_COLOR[priority],
        "description": "\n".join(lines),
        "footer": {"text": f"{issue['shortId']} · {assignee}"},
    }


def build_payload(project_id: str, issues: list[dict], hours: int) -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    # With statsPeriod set, top-level firstSeen is clipped to the window; lifetime has the real one.
    new_ids = {
        i["id"]
        for i in issues
        if datetime.fromisoformat(
            i.get("lifetime", {}).get("firstSeen") or i["firstSeen"]
        )
        >= cutoff
    }
    issues = sorted(
        issues,
        key=lambda i: (PRIORITY_RANK[priority_of(i)], int(i["count"])),
        reverse=True,
    )
    events = sum(int(i["count"]) for i in issues)
    high = sum(1 for i in issues if priority_of(i) == "high")
    sentry_url = (
        f"https://{SENTRY_ORG}.sentry.io/issues/"
        f"?project={project_id}&statsPeriod={hours}h&sort=freq"
    )

    summary_lines = [
        f"- **Events:** {events}",
        f"- **Issues:** {len(issues)}",
        f"- **First seen today:** {len(new_ids)}",
    ]
    hidden = len(issues) - MAX_ISSUE_CARDS
    if hidden > 0:
        summary_lines += [
            "",
            f"Showing top {MAX_ISSUE_CARDS}. [{hidden} more in Sentry]({sentry_url})",
        ]
    if not issues:
        summary_color = COLOR_GREEN
    else:
        summary_color = COLOR_RED if high else COLOR_ORANGE

    cards = [
        {
            "title": f"Summary · last {hours}h",
            "url": sentry_url,
            "color": summary_color,
            "description": "\n".join(summary_lines),
        }
    ]
    cards += [
        build_issue_card(i, i["id"] in new_ids, hours) for i in issues[:MAX_ISSUE_CARDS]
    ]
    cards[-1]["timestamp"] = datetime.now(timezone.utc).isoformat()

    # The message text is the only part shown in push notifications.
    if issues:
        headline = (
            f"**{events} errors** from {len(issues)} "
            f"{'issue' if len(issues) == 1 else 'issues'} in the last {hours}h"
            f" · {high} high priority · {len(new_ids)} first seen today"
        )
    else:
        headline = f"No errors in the last {hours}h"

    role_id = os.environ.get("DISCORD_ROLE_ID")
    return {
        "content": f"<@&{role_id}> {headline}" if role_id else headline,
        "embeds": cards,
        "allowed_mentions": {"roles": [role_id]} if role_id else {"parse": []},
    }


def post_to_discord(webhook_url: str, payload: dict) -> None:
    request = urllib.request.Request(
        webhook_url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "sentry-digest/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            logger.info(f"[post_to_discord] Posted | status: {response.status}")
    except urllib.error.HTTPError as error:
        logger.error(
            f"[post_to_discord] Discord rejected the message | status: {error.code}"
            f" | body: {error.read().decode()}"
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--project", required=True, help="Sentry project slug")
    parser.add_argument("--hours", type=int, default=24, help="look-back window")
    parser.add_argument(
        "--send", action="store_true", help="post to Discord (default: dry run)"
    )
    args = parser.parse_args()

    if not os.environ.get("SENTRY_AUTH_TOKEN"):
        sys.exit("SENTRY_AUTH_TOKEN is not set")
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if args.send and not webhook_url:
        sys.exit("DISCORD_WEBHOOK_URL is not set")

    project_id, issues = list_issues(args.project, args.hours)
    logger.info(
        f"[main] Fetched issues | project: {args.project} | count: {len(issues)}"
    )
    payload = build_payload(project_id, issues, args.hours)

    if args.send:
        post_to_discord(webhook_url, payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
