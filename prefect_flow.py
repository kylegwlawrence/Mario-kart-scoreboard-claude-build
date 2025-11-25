from prefect import flow, task
from dotenv import load_dotenv
import sys
from pathlib import Path
from src.slack_utils import SlackPoller, SlackHandler
from src.race_series import RaceSeries

from src.orchestrator import OCRProcessor

@task
def poll_slack_for_init_message(*args):
    poller = SlackPoller(*args)
    return poller.poll()

@task
def poll_slack_for_next_image(*args):
    poller = SlackPoller(*args)
    return poller.poll()
 
@task
def init_race_series(names:dict, num_races: int, config_path:str, debug=True):
    race_series = RaceSeries(names, num_races, config_path, debug)
    return race_series

@task
def add_race(race_series, image_path, race_number) -> None:
    race_series.add_scoreboard_image(image_path, race_number)

@task
def send_post_race_summary():
    pass

@task
def send_post_series_summary():
    pass

@task
def send_lifetime_summary():
    pass

@task
def image_flow():
    add_race()
    send_post_race_summary()
    send_post_series_summary()
    send_lifetime_summary()

@flow
def main_flow():
    poll_result = poll_slack_for_init_message()
    if poll_result:
        race_series = init_race_series()
        poll_slack_for_next_image() # returns True if there is an image in the channel
        resuts = race_series.add_race(race_series)
        
# on a poll when race initialized and first added, re-poll for next image
    # this means the poll searches for the init message
    # then polls for images
# until all races have been polled
# then return to polling for an init

## wait for next image
    ### send "waiting" message via bot, incude some insults (use sleep talking man)
    ### repeat ingestion of new image file
    ### repeat apply preprocessing and ocr
    ### repeat generate outputs

## apply post-series analytics
    ### merge predictions data
    ### do comparative analysis
        #### share of total points - series, and vs lifetime
        #### simple summary stats for lifetime
        #### n races won
        #### average place
        #### fav character

## send post-series message with analytics summary
    
##### how to loop through the races as the images are submitted? 
##### when ready for the next image, slack sends a message that it is waiting for race N 
#### and how to end the series?