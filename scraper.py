import requests
import json
import pickle
import time
import string
from datetime import datetime
from os.path import exists

flairs = ['left', 'right', 'lib', 'auth', 'centrist', 'authright', 'libright', 'libleft', 'authleft']
user_set_file = "Data/data.pickle"
glob_file = "Data/glob.pickle"
processed_users_file = "Data/processed_users.pickle"
size = 500
num_users = 60000


def load_file(file):
    """
    Takes a pickle filename to be loaded into a variable.
    :param file: filename/path to a pickle file to be loaded
    :return: variable loaded from specified pickle file, None if load fails
    """

    # check if file exists
    if exists(file):
        # open it up
        with open(file, 'rb') as fp:
            # set the variable to be returned
            var = pickle.load(fp)
        return var
    # return None if it doesn't exist
    return None


def write_file(var, file):
    """
    Writes a variable to a pickle file.
    :param var: variable to be written
    :param file: filename/path to file to be written
    :return: None
    """
    # open it up
    with open(file, 'wb') as fp:
        # write it
        pickle.dump(var, fp)
    return


def fmt(flair):
    """
    Formats a users flair to match the list of flairs.
    :param flair: users flair
    :return: matched flair
    """

    # check if flair is empty and longer than 1 character
    if flair is not None and len(flair) > 1:
        # split flair by colon, take latter half, set as lowercase
        flair = flair.split(':')[1].lower()

        # check if flair is edge case, fix if so
        if flair[0] == 'c':
            return 'centrist'
        if flair[-1] == '2':
            return 'libright'

        # return matched flair
        return flair
    else:
        # return None if flair is not usable
        return None


def get_subreddit_data(before, prev_wait=0.0):
    """
    Returns a json object representing *size* comments from subreddit before *before* param,
    handles failure by increasing wait-times.
    :param before: time in seconds after epoch to check for comments before
    :param prev_wait: length of time waited at previous call, defaults to 0
    :return: json object representing comments from subreddit
    """

    # calculate new_wait time exponentially
    new_wait = prev_wait ** 1.5
    if prev_wait <= 1:
        new_wait = 1.1

    # attempt request
    try:
        url = f'https://api.pushshift.io/reddit/comment/search/?size={size}&before={round(before)}' \
              f'&subreddit=PoliticalCompassMemes&filter=id,author,author_flair_text,created_utc'
        print("Requesting new comments...", end='')
        response = requests.get(url, timeout=2)

        # check if request succeeded
        if response:
            # load relevant data and return it
            data = json.loads(response.text)
            return data['data']

        # request failed due to bad response
        else:
            # wait for calculated amount of time
            print(f" got HTTP {response.status_code} error. Trying again after {round(new_wait, 1)} seconds.")
            time.sleep(new_wait)

            # recursively call self with previously calculated wait time
            return get_subreddit_data(before, prev_wait=new_wait)

    # request failed due to timeout
    except requests.exceptions.ReadTimeout:
        # wait for calculated amount of time
        print(f" Timed out. Trying again after {round(new_wait, 1)} seconds.")
        time.sleep(new_wait)

        # recursively call self with previously calculated wait time
        return get_subreddit_data(before, prev_wait=new_wait)


def get_user_comments(author, prev_wait=0.0):
    """
    Returns a list of dictionaries representing a user's comments and scores. Handles failure by increasing wait-times.
    :param author: User to query
    :param prev_wait: length of time waited at previous call, defaults to 0
    :return: list of dictionaries representing a user's comments and scores, a flattened string of all comments
    """

    # calculate new_wait time exponentially
    new_wait = prev_wait ** 1.5
    if prev_wait <= 1:
        new_wait = 1.1
    print("Requesting new comment history...", end='')

    # attempt request
    try:
        url = f'https://api.pushshift.io/reddit/comment/search/?size=150&author={author}&filter=body,score'
        response = requests.get(url, timeout=2)

        # check if request succeeded
        if response:
            # load relevant data
            user_comments = json.loads(response.text)['data']

            # begin cleaning data
            cleaned = []
            for user_comment in user_comments:
                # check if comment includes a link, don't include if so
                # helps to reduce bot comments and unknown tokens
                if 'https://' not in user_comment['body'] and 'http://' not in user_comment['body']:
                    # remove characters that are not letters
                    new_body = ''.join(filter(lambda x: x in string.printable and x not in
                                              ',./?";\'`:;!@#$%^&*()_+1234567890-=\\|}{[]',
                                              user_comment['body'])).replace('\n', ' ')

                    # add comment to list of cleaned comments, include score
                    cleaned.append({'body': new_body,
                                    'score': user_comment['score']})

            # flatten cleaned comments
            user_flat = ' '.join([cmt['body'] for cmt in cleaned])

            # return cleaned comments and flattened string
            return cleaned, user_flat

        # request failed due to bad response
        else:
            # wait for calculated amount of time
            print(f" got HTTP {response.status_code} error. Trying again after {round(new_wait, 1)} seconds.")
            time.sleep(new_wait)

            # recursively call self with previously calculated wait time
            return get_user_comments(author, prev_wait=new_wait)

    # request failed due to no response
    except requests.exceptions.ReadTimeout:
        # wait for calculated amount of time
        print(f" Timed out. Trying again after {round(new_wait, 1)} seconds.")
        time.sleep(new_wait)

        # recursively call self with previously calculated wait time
        return get_user_comments(author, prev_wait=new_wait)


if __name__ == "__main__":
    # load cached user_set, glob, and processed_users pickle files
    user_set = load_file(user_set_file)
    glob = load_file(glob_file)
    processed_users = load_file(processed_users_file)

    # set the oldest_time to now, define minimum wait_time
    oldest_time = round(time.time())
    min_wait = .5

    # check if we got a user_set
    if user_set:
        # set oldest_time to oldest timestamp in user_set
        oldest_time = min([a['created_utc'] for a in user_set.values()])
    else:
        # set user_set to an empty dictionary
        user_set = dict()

    # check if we didn't get a glob
    if not glob:
        # set glob to a dictionary with predefined keys
        glob = {'text': [], 'scores': [], 'labels': []}

    # check if we didn't get a processed_users
    if not processed_users:
        # set processed_users to an empty set
        processed_users = set()

    # set raw_comments to an empty list
    raw_comments = []

    # check if we have enough users
    if len(user_set) < num_users:
        # set raw_comments to subreddit data
        print("Beginning initial query for comments.")
        raw_comments = get_subreddit_data(oldest_time)

    # prepare variables for loop
    added_users = 0
    new_users = 0
    queried_comments = 0

    # while there are entries in the data, and we don't have enough users
    while len(raw_comments) > 0 and len(user_set) < num_users:
        # start timing time between requests
        init_time = time.time()
        print(f" Response received, adding users from {len(raw_comments)} comments from before "
              f"{datetime.fromtimestamp(oldest_time)}, currently have {len(user_set)}...", end='')

        # iterate through all comments
        for comment in raw_comments:
            # check if we've added user yet
            if comment['author'] not in user_set:
                # add new dictionary to user_set with key as user's username
                user_set[comment['author']] = {'created_utc': comment['created_utc'],
                                               'id': comment['id'],
                                               'author_flair_text': fmt(comment['author_flair_text'])}

                # count users added from this data
                added_users += 1

        # count users added since last cache and reset users added from this data
        new_users += added_users
        print(f" Done, added {added_users} users.")
        added_users = 0

        # check if users added since last cache > 100
        if new_users >= 100:
            # update user_set_file and reset counter
            write_file(user_set, user_set_file)
            new_users = 0

        # find new oldest_time by last comment in data
        oldest_time = raw_comments[-1]['created_utc']

        # calculate time to wait based on min_wait time and sleep accordingly
        cur_time = time.time() - init_time
        wait_time = min_wait - cur_time
        if cur_time < min_wait:
            time.sleep(wait_time)

        # get next set of comments and iterate
        raw_comments = get_subreddit_data(oldest_time)

    print(f"Finished with {len(user_set)} total users. Proceeding to comment collection.")

    # prepare variables for loop
    new_users = 0
    queried_users = 0
    init_time = time.time()

    for user in user_set:
        # get flair of user
        user_flair = user_set[user]['author_flair_text']

        # check if flair is acceptable and user has not been added
        if user_flair in flairs and user not in processed_users:
            # calculate time to wait based on wait_time and sleep accordingly
            cur_time = time.time() - init_time
            wait_time = min_wait - cur_time
            if cur_time < min_wait:
                time.sleep(wait_time)

            # get set of user's comments and start timing time between requests
            comments, flat = get_user_comments(user)
            init_time = time.time()
            print(f" Response received, adding comment history, currently have {len(glob['text'])}...",
                  end='')

            # check if there are any comments
            if len(comments) > 0:
                # add data to glob, record user in processed users
                glob['text'].append(flat)
                glob['scores'].append(round(sum(c['score'] for c in comments) / len(comments), 1))
                glob['labels'].append(flairs.index(user_flair))
                processed_users.add(user)

                # count users added since last cache
                new_users += 1
            print(f" Done!")

        # check if users added since last cache > 100
        if new_users >= 100:
            # update glob_file and processed_users_file and reset counter
            write_file(glob, glob_file)
            write_file(processed_users, processed_users_file)
            new_users = 0

    print(f"Finished with {len(glob['text'])} comment histories.")
