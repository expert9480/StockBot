# import smtplib
# from dotenv import load_dotenv
# import os
# import datetime

# load_dotenv()

# server = smtplib.SMTP( "smtp.gmail.com", 587 )
# server.starttls()

# #below password should be from Google Account > Security > App Passwords  NOT the password for your google account
# server.login(os.getenv('gmailAccount'), os.getenv('gmailAppPassword') )
# from_mail = os.getenv('gmailAccount')
 
# def sendMessage(body1, subject):
#     body = str(body1)
#     to = os.getenv('gmailAccount')
#     message = ("From: %s\r\n" % from_mail + "To: %s\r\n" % to + "Subject: %s\r\n" % subject + "\r\n" + body)
#     server.sendmail(from_mail, to, message)

# #class Message:
# def sendReport():
#     time = datetime.datetime.now()
#     time = time.strftime("%m-%d %H:%M")
#     subject = 'Report: ' + str(time)
#     body = 'report'
#     sendMessage(body, subject)

# sendReport()

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
    # App Password (Google Account > Security > App Passwords)
    server.login(os.getenv("gmailAccount"), os.getenv("gmailAppPassword"))
    server.sendmail(from_mail, to, message)
    server.quit()


def sendReport(body="report"):
    time = datetime.datetime.now()
    time = time.strftime("%m-%d %H:%M")
    subject = "Report: " + str(time)
    sendMessage(body, subject)
