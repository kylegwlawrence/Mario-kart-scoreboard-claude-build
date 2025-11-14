import requests
import os
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


# make a file downloader class so we can initiate load_dotenv on class creation
load_dotenv()

def get_new_file_id(previous_files: list, current_files: list) -> str:
    """
    Takes in two lists of files - previous_files_list and current_files_list and determines the missing files in the previous list, if there are any.
    If the two lists are equivalent (ie. no new files), then returns None
    """
    if set(previous_files)!=set(current_files): 
        new_files = list(set(current_files) - set(previous_files))
        if len(new_files)==1:
            file_id = new_files[0]["id"] # assume our new file is the first element in the list
            return file_id
        elif len(new_files)==0:
            raise ValueError(f"There was an error detecting a new file, even though the files between the last loop and now are different.")
        else:
            raise ValueError(f"There is more than one new file in the channel. These are the new files: {[file for file in new_files]}")
    else:
        return None
        
def get_file_metadata(file_id: str, bot_token:str =os.getenv("BOT_TOKEN")) -> dict:
    """
    Use the file's id to get the rest of it's metadata
    """
    
    with WebClient(bot_token) as client:
        try:
            response = client.files_info(file=file_id)
            if response["ok"]:
                return response
            else:
                print(f"Error getting file info: {response['error']}")
        except Exception as e:
            print(f"An error occurred: {e}")
      
      
def download_file(file_metadata:dict, output_path=None, bot_token:str =os.getenv("BOT_TOKEN")) -> None:
    """
    Use a private url found in the file's metadata to download the file
    """
    try:
        headers = {'Authorization': f'Bearer {bot_token}'}
        download_response = requests.get(file_metadata["url_private"], headers=headers, stream=True)
        download_response.raise_for_status()  # Raise an exception for bad status codes

        # check if it is an heic - does slack send PNGs or JPEGs?
        file_name = f"{file_metadata["file"]}.heic"
        with open(f"{file_name}.heic", 'wb') as f: # put output path here?
            for chunk in download_response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File '{file_name}.heic' downloaded successfully.")
        return output_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        
def list_current_files(channel_id: str=os.getenv("CHANNEL_ID"), bot_token: str=os.getenv("BOT_TOKEN")) -> list:
    """
    Get a list of the files currently stored in the channel
    """
    with WebClient(bot_token) as client: 
        try:
            response = client.files_list(channel=channel_id)
            if response["files"] is not None:
                return response["files"]
            else:
                return None
        except SlackApiError as e:
            print(f"Error checking for files: {e}")