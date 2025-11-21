from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from custom_logger import get_custom_logger
import logging
import os
import requests

class SlackBase:

    def __init__(self, channel_id: str, bot_token: str):
        self.channel_id = channel_id
        self.bot_token = bot_token
        self.logger = get_custom_logger(__name__)
        self._test_auth()

    def _test_auth(self):
        """
        Tests authorization and assigns self.bot_user_id
        """
        try:
            self.logger.info("Testing Slack auth...")
            client = WebClient(self.bot_token)
            auth_test = client.auth_test()
            self.bot_user_id = auth_test["user_id"]
            self.logger.info(f"Authentication successful.")
        except SlackApiError as e:
            self.logger.exception(e)
            raise

class SlackPoller(SlackBase):
    
    def _poll_for_one_file(self) -> bool:
        """
        Polls a slack channel to check for the existence of one file in the channel
        
        Returns:
        - True if there is one file
        - False if there are no files
        - Exception for existence of more than one file in the channel
        - Exception for other Slack API errors
        """
        client = WebClient(self.bot_token)
        try:
            response = client.files_list(channel=self.channel_id)
            if len(response["files"]) == 1:
                self.logger.info("Poll successful. Found one file in Slack channel.")
                return True
            elif len(response["files"])==0 or response["files"] is None:
                self.logger.info("Poll successful. No files in Slack channel")
                return False
            else:
                self.logger.exception(f"More than one file was found in slack: {e}")
                raise
        except SlackApiError as e:
            self.logger.exception(e)
            raise
        
    def poll(self) -> bool:
        """
        Public method to poll slack
        """
        result = self._poll_for_one_file()
        return result
    
class SlackHandler(SlackBase):

    def _get_file_url(self) -> None:
        """
        Get the private url of the file in the channel. 
        Assumption: only 0 or 1 files can be in the channel. Any more throws and errors
        """
        client = WebClient(self.bot_token)
        try:
            response = client.files_list(channel=self.channel_id)
            if len(response["files"])==1:
                self.logger.info(f"File retrieval successful. File: {response['files'][0]['name']}")
                self.file_name = response["files"][0]["name"] # this gives us name with extension
                self.url_private = response["files"][0]["url_private"]
                self.file_id = response["files"][0]["id"]
            elif len(response["files"])==0 or response["files"] is None:
                self.logger.info("File retrieval successful. No files in Slack channel")
                return None
            elif len(response["files"])>1:
                self.logger.exception("There is more than one file in the slack channel. There can only be 0 or 1 file in the channel.")
                raise
        except SlackApiError as e:
            self.logger.exception(e)
            raise
        
    def download_file(self, output_dir:str) -> None:
        """
        Use a private url to download the file
        """
        self._get_file_url()
        output_path = f"{output_dir}/{self.file_name}"
        try:
            headers = {'Authorization': f'Bearer {self.bot_token}'}
            download_response = requests.get(self.url_private, headers=headers, stream=True)
            download_response.raise_for_status()  # Raises an exception for bad status codes
            with open(f"{output_path}", 'wb') as f:
                for chunk in download_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            self.logger.info(f"File download successful. File saved to: {output_path}")
        except requests.exceptions.RequestException as e:
            self.logger.exception(e)
            raise
        
    def _upload_file(self, file_path:str, title:str, initial_comment:str):
        """
        Upload a file to the slack channel
        """
        client = WebClient(self.bot_token)
        try:
            self.new_file = client.files_upload_v2(
            channel=self.channel_id,
            title=title,
            file=file_path,
            initial_comment=initial_comment)
            self.logger.info(f"File '{title}' uploaded successfully to Slack")
        except SlackApiError as e:
            self.logger.exception(e)
            raise
    
    def _share_file(self):
        """
        An uploaded file stills needs to be shared with the channel, so this is executed after uploading a file.
        """
        client = WebClient(self.bot_token)
        try:
            self.file_url = self.new_file.get("file").get("permalink")
            self.new_message = client.chat_postMessage(
            channel=self.channel,
            text=f"Here is the file url: {self.file_url}")
            self.logger.info(f"File shared successfully with URL: {self.file_url}")
        except SlackApiError as e:
            self.logger.exception(e)
            raise
    
    def publish_file(self, file_path) -> str:
        """
        Uploads a file and shares it with the channel
        """
        self._upload_file(file_path)
        self._share_file()
        return self.file_url
    
    def delete_file(self):
        """
        Delete a file from the slack channel.
        Required after a new file has been downloaded. Since we always want to grab the newest file, instead of searching through the files' metadata, we will always know the one file in the channel is all we need
        """
        client = WebClient(self.bot_token)
        try:
            client.files_delete(self.file_id)
            self.logger.info(f"File '{self.file_id}' deleted successfully from Slack")
        except SlackApiError as e:
            self.logger.exception(e)
            raise
    
    def send_message(self, message):
        client = WebClient(self.bot_token)
        try:
            response = client.chat_postMessage(
                channel=self.channel_id,
                text=message
            )
            self.logger.info("Message sent successfully to Slack channel")
            return response
        except SlackApiError as e:
            # You will get a SlackApiError if "ok" is False
            self.logger.exception(e)
            raise
            
if __name__ == "__main__":
    load_dotenv()
    
    poller = SlackPoller(os.getenv("CHANNEL_ID"), os.getenv("BOT_TOKEN"))
    result = poller.poll()
    
    handler = SlackHandler(os.getenv("CHANNEL_ID"), os.getenv("BOT_TOKEN"))
    handler.download_file("jpgs")