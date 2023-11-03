from __future__ import print_function
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime
from pygit2 import Repository

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# The ID and range of a sample spreadsheet.
SAMPLE_SPREADSHEET_ID = "1a0KyxWqOCC_sNdxwQ4a0BAZvCWaSjDEUacTW0f85w2M"
SAMPLE_RANGE_NAME = 'Foglio1!A:P'


def publish(data):
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('sheets', 'v4', credentials=creds)

        # Call the Sheets API
        sheet = service.spreadsheets()
        values = data # Modify this list with the data you want to write

        # Find the last written row in the specified range
        result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID, range=SAMPLE_RANGE_NAME).execute()
       
        last_row = len(result.get('values', [])) + 1

        # Append the data to the next empty row
        range_to_append = f'Foglio1!A{last_row}:A{last_row}'
        body = {
            'values': values,
            'majorDimension': 'ROWS'
        }
        request = sheet.values().append(
            spreadsheetId=SAMPLE_SPREADSHEET_ID,
            range=range_to_append,
            valueInputOption='USER_ENTERED',
            body=body
        )
        response = request.execute()
        print('Data written successfully.')

    except HttpError as err:
        print(err)

def push_results(args, total_acc, task_accuracy, accuracy_e, accuracy_taw):
    #keys = [k for k in vars(args)]+["test accuracy", "task accuracy"]
    blacklist = ["cuda","device", "n_classes", "classes_per_exp", "details", "num_workers", "log_every", "plot_gradients_of_layer", "distill_loss"]
    repo = Repository(".")
    branch_name = repo.head.name.split("/")[-1]
    last_commit_id = repo[repo.head.target].hex
    date_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    values = [str(vars(args)[k]) for k in vars(args) if k not in blacklist] + [total_acc.item(), accuracy_e.item(), accuracy_taw.item(), task_accuracy.item()] + [date_time, branch_name, last_commit_id,  vars(args)["details"]]
    publish([values])
