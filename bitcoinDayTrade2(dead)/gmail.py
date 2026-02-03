# import smtplib
# from dotenv import load_dotenv
# import os
# import datetime

# load_dotenv()

# from_mail = os.getenv("gmailAccount")


# def sendMessage(body1, subject):
#     body = str(body1)
#     to = os.getenv("gmailAccount")

#     message = (
#         ("From: %s\r\n" % from_mail)
#         + ("To: %s\r\n" % to)
#         + ("Subject: %s\r\n" % subject)
#         + "\r\n"
#         + body
#     )

#     server = smtplib.SMTP("smtp.gmail.com", 587)
#     server.starttls()
#     server.login(os.getenv("gmailAccount"), os.getenv("gmailAppPassword"))
#     server.sendmail(from_mail, to, message)
#     server.quit()


# def sendReport(body: str, subject_prefix: str = "Report"):
#     time = datetime.datetime.now().strftime("%m-%d %H:%M")
#     subject = f"{subject_prefix}: {time}"
#     sendMessage(body, subject)

import smtplib
from dotenv import load_dotenv
import os
import datetime

load_dotenv()

_SMTP_HOST = "smtp.gmail.com"
_SMTP_PORT = 587

def _get_server():
    server = smtplib.SMTP(_SMTP_HOST, _SMTP_PORT, timeout=20)
    server.starttls()
    server.login(os.getenv("gmailAccount"), os.getenv("gmailAppPassword"))
    return server

def sendReport(body: str, subject_prefix: str = "Report"):
    """
    Sends an email to gmailAccount (self) using app password.
    """
    from_mail = os.getenv("gmailAccount")
    to_mail = os.getenv("gmailAccount")
    now = datetime.datetime.now().strftime("%m-%d %H:%M")
    subject = f"{subject_prefix}: {now}"

    message = (
        f"From: {from_mail}\r\n"
        f"To: {to_mail}\r\n"
        f"Subject: {subject}\r\n"
        f"\r\n"
        f"{body}"
    )

    server = None
    try:
        server = _get_server()
        server.sendmail(from_mail, [to_mail], message)
    finally:
        try:
            if server is not None:
                server.quit()
        except Exception:
            pass
