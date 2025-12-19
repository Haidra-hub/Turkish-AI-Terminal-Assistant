"""
Google Sheets and Drive Integration Module
Provides functionality for reading/writing to Google Sheets and managing Google Drive files
"""

import os
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from google.oauth2.credentials import Credentials as UserCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth import default
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging

logger = logging.getLogger(__name__)

# Google API Scopes
SHEETS_SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive']
ALL_SCOPES = SHEETS_SCOPES + DRIVE_SCOPES


class GoogleConnector:
    """
    Connector for Google Sheets and Drive APIs
    Supports both service account and OAuth2 authentication
    """

    def __init__(self, credentials_path: Optional[str] = None, token_path: Optional[str] = None):
        """
        Initialize Google Connector

        Args:
            credentials_path: Path to service account JSON or OAuth2 credentials file
            token_path: Path to store OAuth2 token (for user authentication)
        """
        self.credentials_path = credentials_path or os.getenv('GOOGLE_CREDENTIALS_PATH')
        self.token_path = token_path or os.getenv('GOOGLE_TOKEN_PATH', 'token.pickle')
        self.credentials = None
        self.sheets_service = None
        self.drive_service = None
        self._authenticate()

    def _authenticate(self) -> None:
        """Authenticate with Google APIs using available credentials"""
        try:
            # Try to use service account credentials first
            if self.credentials_path and os.path.exists(self.credentials_path):
                self.credentials = self._load_service_account_credentials()
                logger.info("Authenticated using service account credentials")
            # Fall back to OAuth2 authentication
            elif os.path.exists(self.token_path):
                self.credentials = self._load_oauth2_credentials()
                logger.info("Authenticated using OAuth2 credentials")
            else:
                # Try default application credentials
                self.credentials, _ = default(scopes=ALL_SCOPES)
                logger.info("Authenticated using default application credentials")

            if self.credentials:
                self.sheets_service = build('sheets', 'v4', credentials=self.credentials)
                self.drive_service = build('drive', 'v3', credentials=self.credentials)
                logger.info("Google services initialized successfully")
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise

    def _load_service_account_credentials(self) -> Credentials:
        """Load service account credentials from JSON file"""
        try:
            return Credentials.from_service_account_file(
                self.credentials_path,
                scopes=ALL_SCOPES
            )
        except Exception as e:
            logger.error(f"Failed to load service account credentials: {str(e)}")
            raise

    def _load_oauth2_credentials(self) -> UserCredentials:
        """Load OAuth2 credentials from pickle file"""
        try:
            with open(self.token_path, 'rb') as token:
                creds = pickle.load(token)
                if creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                return creds
        except Exception as e:
            logger.error(f"Failed to load OAuth2 credentials: {str(e)}")
            raise

    def authenticate_oauth2(self, client_secrets_file: str) -> None:
        """
        Perform OAuth2 authentication flow for user credentials

        Args:
            client_secrets_file: Path to OAuth2 client secrets JSON file
        """
        try:
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_file,
                scopes=ALL_SCOPES
            )
            self.credentials = flow.run_local_server(port=0)

            # Save the credentials for future use
            with open(self.token_path, 'wb') as token:
                pickle.dump(self.credentials, token)

            self.sheets_service = build('sheets', 'v4', credentials=self.credentials)
            self.drive_service = build('drive', 'v3', credentials=self.credentials)
            logger.info("OAuth2 authentication successful")
        except Exception as e:
            logger.error(f"OAuth2 authentication failed: {str(e)}")
            raise

    # ==================== Google Sheets Methods ====================

    def read_sheet(
        self,
        spreadsheet_id: str,
        range_name: str = 'Sheet1'
    ) -> List[List[Any]]:
        """
        Read data from a Google Sheet

        Args:
            spreadsheet_id: ID of the spreadsheet
            range_name: Range to read (e.g., 'Sheet1', 'Sheet1!A1:C10')

        Returns:
            List of lists containing the sheet data
        """
        try:
            result = self.sheets_service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=range_name
            ).execute()
            return result.get('values', [])
        except HttpError as e:
            logger.error(f"Failed to read sheet: {str(e)}")
            raise

    def write_sheet(
        self,
        spreadsheet_id: str,
        range_name: str,
        values: List[List[Any]],
        input_option: str = 'USER_ENTERED'
    ) -> Dict[str, Any]:
        """
        Write data to a Google Sheet

        Args:
            spreadsheet_id: ID of the spreadsheet
            range_name: Range to write to (e.g., 'Sheet1!A1')
            values: List of lists containing data to write
            input_option: How to interpret input (USER_ENTERED or RAW)

        Returns:
            Response from the API
        """
        try:
            body = {
                'values': values
            }
            result = self.sheets_service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption=input_option,
                body=body
            ).execute()
            logger.info(f"Written {result.get('updatedCells')} cells to sheet")
            return result
        except HttpError as e:
            logger.error(f"Failed to write to sheet: {str(e)}")
            raise

    def append_sheet(
        self,
        spreadsheet_id: str,
        range_name: str,
        values: List[List[Any]],
        input_option: str = 'USER_ENTERED'
    ) -> Dict[str, Any]:
        """
        Append data to a Google Sheet

        Args:
            spreadsheet_id: ID of the spreadsheet
            range_name: Range to append to (e.g., 'Sheet1!A:A')
            values: List of lists containing data to append
            input_option: How to interpret input (USER_ENTERED or RAW)

        Returns:
            Response from the API
        """
        try:
            body = {
                'values': values
            }
            result = self.sheets_service.spreadsheets().values().append(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption=input_option,
                body=body
            ).execute()
            logger.info(f"Appended {result.get('updates', {}).get('updatedRows', 0)} rows to sheet")
            return result
        except HttpError as e:
            logger.error(f"Failed to append to sheet: {str(e)}")
            raise

    def clear_sheet(
        self,
        spreadsheet_id: str,
        range_name: str
    ) -> Dict[str, Any]:
        """
        Clear data from a Google Sheet

        Args:
            spreadsheet_id: ID of the spreadsheet
            range_name: Range to clear (e.g., 'Sheet1!A1:C10')

        Returns:
            Response from the API
        """
        try:
            result = self.sheets_service.spreadsheets().values().clear(
                spreadsheetId=spreadsheet_id,
                range=range_name
            ).execute()
            logger.info(f"Cleared {result.get('clearedCells')} cells")
            return result
        except HttpError as e:
            logger.error(f"Failed to clear sheet: {str(e)}")
            raise

    def batch_update_sheet(
        self,
        spreadsheet_id: str,
        requests: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform batch updates on a Google Sheet

        Args:
            spreadsheet_id: ID of the spreadsheet
            requests: List of update requests

        Returns:
            Response from the API
        """
        try:
            body = {'requests': requests}
            result = self.sheets_service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body=body
            ).execute()
            logger.info(f"Batch updated sheet with {len(requests)} requests")
            return result
        except HttpError as e:
            logger.error(f"Failed to batch update sheet: {str(e)}")
            raise

    def get_sheet_properties(self, spreadsheet_id: str) -> Dict[str, Any]:
        """
        Get properties of a Google Sheet

        Args:
            spreadsheet_id: ID of the spreadsheet

        Returns:
            Sheet properties dictionary
        """
        try:
            result = self.sheets_service.spreadsheets().get(
                spreadsheetId=spreadsheet_id
            ).execute()
            return result
        except HttpError as e:
            logger.error(f"Failed to get sheet properties: {str(e)}")
            raise

    # ==================== Google Drive Methods ====================

    def list_files(
        self,
        query: Optional[str] = None,
        max_results: int = 10,
        page_token: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        List files from Google Drive

        Args:
            query: Query string to filter files (e.g., "name='test.csv'")
            max_results: Maximum number of results to return
            page_token: Token for pagination

        Returns:
            Tuple of (list of files, next page token)
        """
        try:
            request_args = {
                'spaces': 'drive',
                'pageSize': max_results,
                'fields': 'nextPageToken, files(id, name, mimeType, createdTime, modifiedTime, size)',
            }
            if query:
                request_args['q'] = query
            if page_token:
                request_args['pageToken'] = page_token

            result = self.drive_service.files().list(**request_args).execute()
            return result.get('files', []), result.get('nextPageToken')
        except HttpError as e:
            logger.error(f"Failed to list files: {str(e)}")
            raise

    def search_files(self, filename: str) -> List[Dict[str, Any]]:
        """
        Search for files by name in Google Drive

        Args:
            filename: Name of the file to search for

        Returns:
            List of matching files
        """
        try:
            query = f"name='{filename}' and trashed=false"
            files, _ = self.list_files(query=query, max_results=10)
            return files
        except Exception as e:
            logger.error(f"Failed to search files: {str(e)}")
            raise

    def upload_file(
        self,
        file_path: str,
        folder_id: Optional[str] = None,
        file_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload a file to Google Drive

        Args:
            file_path: Local path to the file
            folder_id: ID of the folder to upload to (optional)
            file_name: Name for the file in Drive (defaults to local filename)

        Returns:
            File metadata
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            file_name = file_name or os.path.basename(file_path)
            
            file_metadata = {'name': file_name}
            if folder_id:
                file_metadata['parents'] = [folder_id]

            media = build('drive', 'v3', credentials=self.credentials).files().create_media(
                body=file_metadata
            )
            
            # Import here to avoid circular imports
            from googleapiclient.http import MediaFileUpload
            media = MediaFileUpload(file_path, resumable=True)
            
            request = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name, webViewLink'
            )
            response = request.execute()
            logger.info(f"Uploaded file: {file_name} (ID: {response.get('id')})")
            return response
        except Exception as e:
            logger.error(f"Failed to upload file: {str(e)}")
            raise

    def download_file(self, file_id: str, output_path: str) -> None:
        """
        Download a file from Google Drive

        Args:
            file_id: ID of the file to download
            output_path: Local path to save the file
        """
        try:
            from googleapiclient.http import MediaIoBaseDownload
            import io

            request = self.drive_service.files().get_media(fileId=file_id)
            fh = io.FileIO(output_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            logger.info(f"Downloaded file to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to download file: {str(e)}")
            raise

    def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """
        Get metadata for a file

        Args:
            file_id: ID of the file

        Returns:
            File metadata
        """
        try:
            result = self.drive_service.files().get(
                fileId=file_id,
                fields='id, name, mimeType, createdTime, modifiedTime, size, webViewLink'
            ).execute()
            return result
        except HttpError as e:
            logger.error(f"Failed to get file info: {str(e)}")
            raise

    def delete_file(self, file_id: str) -> None:
        """
        Delete a file from Google Drive

        Args:
            file_id: ID of the file to delete
        """
        try:
            self.drive_service.files().delete(fileId=file_id).execute()
            logger.info(f"Deleted file: {file_id}")
        except HttpError as e:
            logger.error(f"Failed to delete file: {str(e)}")
            raise

    def create_folder(self, folder_name: str, parent_id: Optional[str] = None) -> str:
        """
        Create a folder in Google Drive

        Args:
            folder_name: Name of the folder
            parent_id: ID of the parent folder (optional)

        Returns:
            ID of the created folder
        """
        try:
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            if parent_id:
                file_metadata['parents'] = [parent_id]

            result = self.drive_service.files().create(
                body=file_metadata,
                fields='id'
            ).execute()
            folder_id = result.get('id')
            logger.info(f"Created folder: {folder_name} (ID: {folder_id})")
            return folder_id
        except HttpError as e:
            logger.error(f"Failed to create folder: {str(e)}")
            raise

    def share_file(
        self,
        file_id: str,
        email: str,
        role: str = 'reader'
    ) -> Dict[str, Any]:
        """
        Share a file with another user

        Args:
            file_id: ID of the file to share
            email: Email address to share with
            role: Role to grant (owner, organizer, fileOrganizer, writer, commenter, reader)

        Returns:
            Permission metadata
        """
        try:
            permission = {
                'type': 'user',
                'role': role,
                'emailAddress': email
            }
            result = self.drive_service.permissions().create(
                fileId=file_id,
                body=permission,
                fields='id'
            ).execute()
            logger.info(f"Shared file {file_id} with {email} as {role}")
            return result
        except HttpError as e:
            logger.error(f"Failed to share file: {str(e)}")
            raise


# Convenience functions for common operations

def read_google_sheet(spreadsheet_id: str, range_name: str = 'Sheet1') -> List[List[Any]]:
    """Quick function to read from a Google Sheet"""
    connector = GoogleConnector()
    return connector.read_sheet(spreadsheet_id, range_name)


def write_google_sheet(
    spreadsheet_id: str,
    range_name: str,
    values: List[List[Any]]
) -> Dict[str, Any]:
    """Quick function to write to a Google Sheet"""
    connector = GoogleConnector()
    return connector.write_sheet(spreadsheet_id, range_name, values)


def upload_to_drive(file_path: str, folder_id: Optional[str] = None) -> Dict[str, Any]:
    """Quick function to upload a file to Google Drive"""
    connector = GoogleConnector()
    return connector.upload_file(file_path, folder_id)
