# adapted from this post: https://stackoverflow.com/a/6270987

from email.mime.text import MIMEText
import json
import logging
import os
import smtplib
from pathlib import Path

logger = logging.getLogger(__name__)

_MAIL_DIR = Path(__file__).resolve().parent
if os.getenv("FLASK_DEBUG") == "1":
    email_list_suffix = "_dev"
else:
    email_list_suffix = ""
_basename = f"notification_emails{email_list_suffix}"
for _name in (f"{_basename}.json", f"{_basename}.json.example"):
    _path = _MAIL_DIR / _name
    if _path.is_file():
        with open(_path) as f:
            default_recipients = json.load(f)
        break
else:
    default_recipients = []


def send_mail(subject, msg, recipients=default_recipients):
    if not recipients:
        logger.info("send_mail skipped: no recipients configured (%s)", subject)
        return
    mime_msg = MIMEText(msg)
    mime_msg["Subject"] = subject
    me = "Egg Count Tester <donotreply@rebeccayang.org>"
    mime_msg["From"] = me
    mime_msg["To"] = ", ".join(recipients)

    # Send the message via our own SMTP server, but don't include the
    # envelope header. Failures here must not crash the caller — on dev
    # boxes there is typically no local SMTP server listening.
    try:
        s = smtplib.SMTP("localhost", 25)
        s.sendmail(me, recipients, mime_msg.as_string())
        s.quit()
    except (OSError, smtplib.SMTPException) as exc:
        logger.warning("send_mail failed (%s): %s", subject, exc)
