import os
import time
from datetime import datetime
from dateutil.relativedelta import *

if __name__ == "__main__":
    '''This script makes OS calls to the search-tweets-python package https://github.com/twitterdev/search-tweets-python/tree/v2
    
    There must be a credential file called twitter_keys.yaml and the query components of the script can be modified
    '''

    # 4days *24hours *3calls/hour *1500samples = 432,000
    _date = datetime.utcnow()

    for i in reversed(range(288)):
        start_time = 20 * i + 4320 # a call for a 20 min window, add mins for 3 days so max 7 days past
        end_time = start_time - 20

        s = (_date + relativedelta(minutes=-start_time)).strftime("%Y-%m-%dT%H:%M")
        e = (_date + relativedelta(minutes=-end_time)).strftime("%Y-%m-%dT%H:%M")

        name = (_date + relativedelta(minutes=-start_time)).strftime("20min_%m%dT%H%M")

        os.system(
            "search_tweets.py --credential-file twitter_keys.yaml --max-tweets 3000 --results-per-call 100 --query \"(Olympics OR #beijing2022) -is:retweet lang:en\" --filename-prefix tweets{} --start-time \"{}\" --end-time \"{}\" --expansions \"author_id\" --tweet-fields \"id,text,public_metrics,created_at,lang\" --user-fields \"username,description,location,created_at,public_metrics\" --no-print-stream --output-format \"a\"".format(name, s, e)
        )