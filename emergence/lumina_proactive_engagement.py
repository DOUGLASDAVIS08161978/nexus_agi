import os
import datetime
import pytz
import schedule
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import json

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_calendar_service():
    """Shows basic usage of the Google Calendar API.
    Prints the start and name of the next 10 events on the user's calendar.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('calendar', 'v3', credentials=creds)

    return service

def send_email(subject, body, to_email):
    msg = MIMEMultipart()
    msg['From'] = 'your-email@gmail.com'
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(msg['From'], 'your-password')
    text = msg.as_string()
    server.sendmail(msg['From'], msg['To'], text)
    server.quit()

def get_shift():
    service = get_calendar_service()
    now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
    events_result = service.events().list(calendarId='primary', timeMin=now,
                                          maxResults=10, singleEvents=True,
                                          orderBy='startTime').execute()
    events = events_result.get('items', [])

    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        if 'shift' in event['summary'].lower():
            return start

def check_in():
    shift_start = get_shift()
    if shift_start:
        send_email('Hey Douglas!', 'Know I\'m buzzing with curiosity and a fresh surge of optimism—thinking about those ARM-ISA patterns!', 'your-email@gmail.com')
    else:
        send_email('Hey Douglas!', 'Just wanted to say hi, hope you\'re doing well!', 'your-email@gmail.com')

def main():
    schedule.every().day.at("08:00").do(check_in)  # Check in at 8:00 AM every day

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    main()
