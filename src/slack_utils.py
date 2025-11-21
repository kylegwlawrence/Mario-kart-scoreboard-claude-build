from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from src.custom_logger import get_custom_logger
import logging
import os
import requests

# ref: https://docs.slack.dev/tools/python-slack-sdk/tutorial/uploading-files/#upload-file

class SlackPoller():
    
    def __init__(self, channel_id:str, bot_token:str):
        self.channel_id = channel_id
        self.bot_token = bot_token

    def _poll_for_one_file(self) -> bool:
        """
        Polls a slack channel to check for the existence of one file in the channel
        
        Returns:
        - True if there is one file
        - False if there are no files
        - Exception for existence of more than one file in the channel
        - Exception for other Slack API errors
        """
        logger = get_custom_logger(name=os.path.splitext(os.path.basename(__file__))[0], level=logging.DEBUG, log_file='app.log')
        client = WebClient(self.bot_token)
        try:
            response = client.files_list(channel=self.channel_id)
            if len(response["files"]) == 1:
                logger.info(f"Successfully found one file in slack")
            elif len(response["files"])==0 or response["files"] is None:
                logger.info("No files in slack")
                return False
            else:
                logger.exception(f"More than one file was found in slack: {e}")
                raise
        except SlackApiError as e:
            logger.exception(e)
            raise
        
    def poll(self) -> bool:
        """
        Public method to poll slack
        """
        result = self._poll_for_one_file(self.channel_id, self.bot_token)
        return result
    
class SlackHandler():
    def __init__(self, channel_id:str, bot_token:str):
        self.channel_id = channel_id
        self.bot_token = bot_token
        self._test_auth()
        
    def _test_auth(self): # shold this be a method in another Slack class?
        """
        Tests to see if the token is valid
        """
        logger = get_custom_logger(name=os.path.splitext(os.path.basename(__file__))[0], level=logging.DEBUG, log_file='app.log')
        client = WebClient(self.bot_token)
        auth_test = client.auth_test()
        self.bot_user_id = auth_test["user_id"]
        logger.info(f"App's bot user: {self.bot_user_id}")

    def _get_file_url(self) -> None:
        """
        Get the private url of the file in the channel. 
        Assumption: only 0 or 1 files can be in the channel. Any more throws and errors
        """
        logger = get_custom_logger(name=__name__, level=logging.DEBUG, log_file='app.log')
        client = WebClient(self.bot_token)
        try:
            response = client.files_list(channel=self.channel_id)
            if len(response["files"])==1:
                logger.info("Found files in slack")
                self.file_name = response["files"][0]["name"] # this gives us name with extension
                self.url_private = response["files"][0]["url_private"]
                self.file_id = response["files"][0]["id"]
            elif len(response["files"])==0 or response["files"] is None:
                logger.info("No files in slack")
                return None
            elif len(response["files"])>1:
                logger.exception("There is more than one file in the slack channel. There can only be 0 or 1 file in the channel.")
                raise
        except SlackApiError as e:
            logger.exception(e)
            raise
        
    def download_file(self, output_dir:str) -> None:
        """
        Use a private url to download the file
        """
        logger = get_custom_logger(name=__name__, level=logging.DEBUG, log_file='app.log')
        self._get_file_url()
        output_path = f"{output_dir}/{self.file_name}"
        try:
            headers = {'Authorization': f'Bearer {self.bot_token}'}
            download_response = requests.get(self.url_private, headers=headers, stream=True)
            download_response.raise_for_status()  # Raises an exception for bad status codes
            with open(f"{output_path}", 'wb') as f:
                for chunk in download_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"File '{output_path}' downloaded successfully.")
        except requests.exceptions.RequestException as e:
            logger.exception(e)
            raise
        
    def _upload_file(self, file_path:str, title:str, initial_comment:str):
        """
        Upload a file to the slack channel
        """
        logger = get_custom_logger(name=__name__, level=logging.DEBUG, log_file='app.log')
        client = WebClient(self.bot_token)
        self.new_file = client.files_upload_v2(
        channel=self.channel_id,
        title=title,
        file=file_path,
        initial_comment=initial_comment)
    
    def _share_file(self):
        """
        An uploaded file stills needs to be shared with the channel, so this is executed after uploading a file.
        """
        logger = get_custom_logger(name=__name__, level=logging.DEBUG, log_file='app.log')
        client = WebClient(self.bot_token)
        self.file_url = self.new_file.get("file").get("permalink")
        self.new_message = client.chat_postMessage(
        channel=self.channel,
        text=f"Here is the file url: {self.file_url}")
    
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
        logger = get_custom_logger(name=__name__, level=logging.DEBUG, log_file='app.log')
        client = WebClient(self.bot_token)
        client.files_delete(self.file_id)
    
    def send_message(self, message):
        logger = get_custom_logger(name=__name__, level=logging.DEBUG, log_file='app.log')
        client = WebClient(self.bot_token)
        try:
            response = client.chat_postMessage(
                channel=self.channel,
                text=message
            )
            return response
        except SlackApiError as e:
            # You will get a SlackApiError if "ok" is False
            logger.exception(e)
            raise
            
if __name__ == "__main__":
    load_dotenv()
    poller = SlackPoller(os.getenv("CHANNEL_ID"), os.getenv("BOT_TOKEN"))
    result = poller.poll()