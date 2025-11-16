import requests
import os
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import time
from custom_logger import get_custom_logger
import logging

def list_files(channel_id: str, bot_token: str) -> list:
    """
    Get a list of the files currently stored in the channel
    """
    logger = get_custom_logger(name=__name__, level=logging.DEBUG, log_file='app.log')
    client = WebClient(bot_token)
    try:
        response = client.files_list(channel=channel_id)
        if len(response["files"])==0 or response["files"] is None:
            logger.info("No files in slack")
            return None
        else:
            logger.info("Found files in slack")
            return response["files"]
    except SlackApiError as e:
        logger.exception(e)
        raise

def find_file_id(previous_files: list, current_files: list) -> str:
    """
    Takes in two lists of file objects and determines the missing files in the first list, if there are any.

    Args:
    - previous_files (list): a list of file objects in slack from a previous timeframe
    - current_files (list): a linewst of file objects in slack right now

    Returns:
    - file_id (str, None): the id of the file missing from the previous timeframe. Returns None if the two lists match
    """
    logger = get_custom_logger(name=__name__, level=logging.DEBUG, log_file='app.log')
    if previous_files is None:
        previous_files = []
    if current_files is None:
        current_files = []

    if len(current_files)>0:

        # Extract file IDs for comparison (dicts are unhashable, so compare IDs instead)
        previous_file_ids = {f["id"] for f in previous_files} if previous_files else set()
        current_file_ids = {f["id"] for f in current_files}

        if previous_file_ids != current_file_ids:
            new_file_ids = list(current_file_ids - previous_file_ids)
            if len(new_file_ids)==1:
                file_id = new_file_ids[0] # assume our new file is the first element in the list
                logger.info(f"File id retrieved: {file_id}")
                return file_id
            elif len(new_file_ids)==0:
                with ValueError(f"There was an error detecting a new file, even though the files between the last loop and now are different.") as e:
                    logger.error(e, exc_info=True)
                    raise
            else:
                with ValueError(f"There is more than one new file in the channel. These are the new files: {new_file_ids}") as e:
                    logger.error(e, exc_info=True)
                    raise
        else:
            logger.info("Previous files and current files are the same")
            return None
    else:
        logger.info("There are no files in the channel")
        return None
        
def get_file_url(file_id: str, bot_token: str) -> tuple:
    """
    Use the file's id to get the image's download URL
    """
    logger = get_custom_logger(name=__name__, level=logging.DEBUG, log_file='app.log')
    client = WebClient(bot_token)
    try:
        response = client.files_info(file=file_id)
        logger.info(f"Retrieved file metadata: {response["file"]["url_private"]}, {response["file"]["name"]}")
        return (response["file"]["url_private"], response["file"]["name"])
    except Exception as e:
        logger.exception(e)
      
def download_file(url:str, output_path:str, bot_token:str) -> str:
    """
    Use a private url found in the file's metadata to download the file and save it to the output path
    """
    logger = get_custom_logger(name=__name__, level=logging.DEBUG, log_file='app.log')
    try:
        headers = {'Authorization': f'Bearer {bot_token}'}
        download_response = requests.get(url, headers=headers, stream=True)
        download_response.raise_for_status()  # Raise an exception for bad status codes

        # check if it is an heic - does slack send PNGs or JPEGs?
        #file_name = f"{file_metadata["file"]}.png"
        with open(f"{file_name}", 'wb') as f: # put output path here?
            for chunk in download_response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File '{file_name}' downloaded successfully.")
        return file_name
    except requests.exceptions.RequestException as e:
        logger.exception(e)
        
def main(env_file_path: str, download_output_dir: str) -> None:
    """
    Continuously checks if there are files in the slack channel, determines if there is a new file, and triggers another process to download the new file. Intended to be always on. Slack channel id and apikey are stored in .env
    
    Args:
    - env_file_path (str): path to environment variables file
    - download_output_dir (str): the directory where the downloaded file will be saved. 
    
    Returns:
    None
    """
    logger = get_custom_logger(name=__name__, level=logging.DEBUG, log_file='app.log')
    logger.info("Slack file agent has started up")
    
    # load env variables
    load_dotenv(env_file_path)
    
    # init the previous files list
    previous_files = None

    # keep the script running
    while True:
        trigger_script = False
        current_files = list_files()
        if current_files is None:
            continue
        elif current_files is not None:
            if previous_files is not None:
                new_file_id = list_files(previous_files, current_files)
                if new_file_id is not None:
                    logger.info("Trigger OCR script")
                    trigger_script = True
                else:
                    continue
            elif previous_files is None:
                new_file_id = current_files[0]["id"] # the first file is still a new file
                if new_file_id is not None:
                    logger.info("Trigger OCR script")
                    trigger_script = True # there is a new file so trigger the OCR script at end of loop
                else:  # there should be a file since we satisified the condition of having at least one file in the current files list. 
                   with ValueError(f"There is no file in the current files list, even though current files list was evaluated as not None earlier in the script.") as e:
                       logging.error(e, exc_info=True)
            new_file_metadata = get_file_url(new_file_id)
            new_file_name = new_file_metadata["name"]
            new_file_path = f"{download_output_dir}/{new_file_name}"
            download_file(new_file_metadata, new_file_path)
             
        if trigger_script:
            try:
                # pass the new_file_path to the OCR script (python program)
                # don't use subprocessing - block this program from continuing until OCR is done
                # point to OCR process and trigger it. Import this as a module?
                pass 
            except Exception as e:
                logger.exception(f"Error triggering script: {e}")
    
        # the current files will be the previous files list for the next loop, even if current files are empty
        previous_files = current_files.copy()
            
        time.sleep(5)
        
if __name__ == "__main__":

    load_dotenv()
    
    # test retrieving file id of new file
    current_files = list_files(os.getenv("CHANNEL_ID"), os.getenv("BOT_TOKEN"))
    previous_files = []
    file_id = find_file_id(previous_files, current_files)
    
    # test getting metadata for file id
    url, file_name = get_file_url(file_id, os.getenv("BOT_TOKEN"))
    
    print(url)
    
    # test downloading file with private url
    downloaded_file_name = download_file(url, "test_slack_download.png", os.getenv("BOT_TOKEN"))
    print(downloaded_file_name)