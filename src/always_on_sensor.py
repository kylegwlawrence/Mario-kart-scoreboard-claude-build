import time
from dotenv import load_dotenv
from new_file_from_slack import (
    list_current_files
    , get_new_file_id
    , get_file_metadata
    , download_file
    )

def main(download_output_dir):
    """
    Continuously checks if there are files in the slack channel, determines if there is a new file, and triggers another process to download the new file. Intended to be always on. Slack channel id and apikey are stored in .env
    
    Args:
    download_output_dir (str): the directory where the downloaded file will be saved. 
    """
    
    # load env variables
    load_dotenv(".env")
    
    # init the previous files list
    previous_files = None

    # keep the script running
    while True:
        trigger_script = False
        
        # query a slack channel for files
        current_files = list_current_files()
        
        # if there are no files in the channel, continue to next loop
        if current_files is None:
            continue
        
        # if there are files in the channel
        elif current_files is not None:
            
            # if we have encountered files before
            if previous_files is not None:
                
                # compare the two lists of files and get the new file id
                new_file_id = get_new_file_id(previous_files, current_files)
                
                if new_file_id is not None:
                    # there is a new file so trigger the OCR script at end of loop
                    trigger_script = True
                else:
                    # continue to the next loop - no new files
                    continue
                
            # if we encounter files for the first time
            elif previous_files is None:
                
                # the first file is still a new file
                new_file_id = current_files[0]["id"]
                
                if new_file_id is not None:
                    # there is a new file so trigger the OCR script at end of loop
                    trigger_script = True
                else:
                    # there should be a file since we satisified the condition of having at least one file in the current files list. 
                    raise ValueError(f"There is no file in the current files list, even though current files list was evaluated as not None earlier in the script.")
                
            # get file's metadata which contains a private url for the file and download
            new_file_metadata = get_file_metadata(new_file_id)
            new_file_name = new_file_metadata["name"]
            new_file_path = f"{download_output_dir}/{new_file_name}.png"
            download_file(new_file_metadata, new_file_path)
             
        if trigger_script:
            # pass the new_file_path to the OCR script (python program)
            # don't use subprocessing - block this program from continuing until OCR is done
            # point to OCR process and trigger it. Import this as a module?
            pass 
    
        # the current files will be the previous files list for the next loop, even if current files are empty
        del previous_files
        previous_files = current_files.copy()
        
        # delete variables to avoid mutability - recreate them in next loop
        del current_files
        del trigger_script
            
        time.sleep(5)