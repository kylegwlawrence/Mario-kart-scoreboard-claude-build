from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from custom_logger import get_custom_logger
import logging
import os
import requests

class SlackBase:

    def __init__(self, channel_id: str, bot_token: str, user_token: str = None):
        self.logger = get_custom_logger(__name__)
        self.channel_id = channel_id
        self.bot_token = bot_token
        self._test_bot_auth()
        if user_token is not None:
            self.user_token = user_token
            self._test_user_auth()


    def _test_bot_auth(self):
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
        
    def _test_user_auth(self):
        """
        Tests authorization and assigns self.bot_user_id
        """
        try:
            self.logger.info("Testing Slack auth...")
            client = WebClient(self.bot_token)
            auth_test = client.auth_test()
            self.user_id = auth_test["user_id"]
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
                error_message = "During polling, found more than one file in the channel. There can only be max one file in the channel. Manually delete the unnecesaary files in the channel."
                logging.error(error_message)
                raise ValueError(error_message)
        except SlackApiError as e:
            self.logger.exception(e)
        
    def poll(self) -> bool:
        """
        Public method to poll slack
        """
        result = self._poll_for_one_file()
        return result
    
class SlackHandler(SlackBase):

    def _get_file_info(self) -> None:
        """
        Get the private url of the file in the channel. 
        Assumption: only 0 or 1 files can be in the channel. Any more throws and errors
        """
        client = WebClient(self.bot_token)
        try:
            response = client.files_list(channel=self.channel_id)
            if len(response["files"])==1:
                self.file_name = response["files"][0]["name"] # this gives us name with extension
                self.url_private = response["files"][0]["url_private"]
                self.file_id = response["files"][0]["id"]
                self.logger.info("Retrieved file info from file in Slack")
            elif len(response["files"])==0 or response["files"] is None:
                self.file_name = None
                self.url_private = None
                self.file_id = None
                self.logger.info("File retrieval successful. No files in Slack channel")
            elif len(response["files"])>1:
                error_message = "There can only be max one file in the channel"
                logging.error(error_message)
                raise ValueError(error_message)
        except SlackApiError as e:
            self.logger.exception(e)
        
    def download_file(self, output_dir:str) -> None:
        """
        Use a private url to download the file, if a file exists
        """
        self._get_file_info()
        if self.file_id is not None:
            output_path = f"{output_dir}/{self.file_name}"
            try:
                headers = {'Authorization': f'Bearer {self.bot_token}'}
                download_response = requests.get(self.url_private, headers=headers, stream=True)
                with open(f"{output_path}", 'wb') as f:
                    for chunk in download_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                self.logger.info(f"File download successful. File saved to: {output_path}")
            except requests.exceptions.RequestException as e:
                self.logger.exception(e)
        else:
            self.logger.info("No files to download")
        
    def _upload_file(self, file_path:str, image_title:str, initial_comment:str = None):
        """
        Upload a file to the slack channel
        Add: prefix on the name to identify bot published files
        """
        client = WebClient(self.bot_token)
        try:
            self.new_file = client.files_upload_v2(
            channel=self.channel_id,
            title=image_title,
            file=file_path,
            initial_comment=initial_comment)
            self.logger.info(f"File '{image_title}' uploaded successfully to Slack")
        except SlackApiError as e:
            self.logger.exception(e)
    
    def _share_file(self):
        """
        An uploaded file stills needs to be shared with the channel, so this is executed after uploading a file.
        Apparently you don't need to share the file since files_upload_v2 makes it viewable in the channel
        """
        client = WebClient(self.bot_token)
        try:
            self.file_url = self.new_file.get("file").get("permalink")
            self.new_message = client.chat_postMessage(
            channel=self.channel_id,
            text=f"Here is the file url: {self.file_url}")
            self.logger.info(f"File shared successfully with URL: {self.file_url}")
        except SlackApiError as e:
            self.logger.exception(e)
    
    def publish_file(self, file_path, image_title, initial_comment) -> str:
        """
        Uploads a file, names it, and adds a comment to the message
        """
        self._upload_file(file_path, image_title, initial_comment)
        #self._share_file()
        #return self.file_url
    
    def delete_file(self) -> None:
        """
        Delete a file from the slack channel.
        Required after a new file has been downloaded. Since we always want to grab the newest file, instead of searching through the files' metadata, we will always know the one file in the channel is all we need
        Must use a USER_TOKEN instead of a BOT_TOKEN - bots cannot delete a file that they don't own.
        """
        self._get_file_info()
        if self.file_id is not None:
            client = WebClient(self.user_token)
            try:
                client.files_delete(file=self.file_id)
                self.logger.info(f"File '{self.file_id}' has been deleted from Slack")
            except SlackApiError as e:
                self.logger.exception(e)
        else:
            self.logger.info("No files to delete")
    
    def send_message(self, message) -> None:
        client = WebClient(self.bot_token)
        try:
            client.chat_postMessage(
                channel=self.channel_id,
                text=message
            )
            self.logger.info("Message sent successfully to Slack channel")
        except SlackApiError as e:
            self.logger.exception(e)
    
if __name__ == "__main__":
    load_dotenv()
    
    handler = SlackHandler(os.getenv("CHANNEL_ID"), os.getenv("BOT_TOKEN"), os.getenv("USER_TOKEN"))
    
    handler.download_file("jpgs")
    
    handler.delete_file()
    
    handler.publish_file(
        file_path="jpgs/IMG_7995.jpg"
        , image_title = "bot_upload"
        , initial_comment="Hey ya fat dink"
        )