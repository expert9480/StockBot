import smtplib
from dotenv import load_dotenv
import os
import datetime

load_dotenv()

from_mail = os.getenv("gmailAccount")


def sendMessage(body1, subject):
    body = str(body1)
    to = os.getenv("gmailAccount")

    message = (
        ("From: %s\r\n" % from_mail)
        + ("To: %s\r\n" % to)
        + ("Subject: %s\r\n" % subject)
        + "\r\n"
        + body
    )

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(os.getenv("gmailAccount"), os.getenv("gmailAppPassword"))
    server.sendmail(from_mail, to, message)
    server.quit()


def sendReport(body: str, subject_prefix: str = "Report"):
    time = datetime.datetime.now().strftime("%m-%d %H:%M")
    subject = f"{subject_prefix}: {time}"
    sendMessage(body, subject)

