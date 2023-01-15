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
num_users = 50000


def load_file(file):
    if exists(file):
        with open(file, 'rb') as fp:
            var = pickle.load(fp)
        return var
    return None


def write_file(var, file):
    with open(file, 'wb') as fp:
        pickle.dump(var, fp)
    return


def fmt(flair):
    if flair is not None and len(flair) > 1:
        flair = flair.split(':')[1].lower()
        if flair[0] == 'c':
            return 'centrist'
        if flair[-1] == '2':
            return 'libright'
        return flair
    else:
        return None


def get_subreddit_data(before, prev_wait):
    new_wait = prev_wait ** 1.5
    if prev_wait <= 1:
        new_wait = 1.1
    try:
        url = f'https://api.pushshift.io/reddit/comment/search/?size={size}&before={round(before)}' \
              f'&subreddit=PoliticalCompassMemes&filter=id,author,author_flair_text,created_utc'
        print("Requesting new comments...", end='')
        response = requests.get(url, timeout=2)
        if response:
            data = json.loads(response.text)
            return data['data']
        else:
            print(f" Got HTTP {response.status_code} error. Trying again.")
            time.sleep(new_wait)
            return get_subreddit_data(before, new_wait)
    except requests.exceptions.ReadTimeout:
        print(f" Timed out. Trying again.")
        time.sleep(new_wait)
        return get_subreddit_data(before, new_wait)


def get_user_comments(author, prev_wait):
    new_wait = prev_wait ** 1.5
    if prev_wait <= 1:
        new_wait = 1.1
    print("Requesting new comment history...", end='')
    try:
        url = f'https://api.pushshift.io/reddit/comment/search/?size=150&author={author}&filter=body,score'
        response = requests.get(url, timeout=4)
        if response:
            user_comments = json.loads(response.text)['data']
            cleaned = []
            for user_comment in user_comments:
                if 'https://' not in user_comment['body'] and 'http://' not in user_comment['body']:
                    new_body = ''.join(filter(lambda x: x in string.printable and x not in
                                              ',./?";\'`:;!@#$%^&*()_+1234567890-=\\|}{[]',
                                              user_comment['body'])).replace('\n', ' ')
                    cleaned.append({'body': new_body,
                                    'score': user_comment['score']})
            user_flat = ' '.join([cmt['body'] for cmt in cleaned])
            return cleaned, user_flat
        else:
            print(f" got HTTP {response.status_code} error. Trying again after {round(new_wait, 1)} seconds.")
            time.sleep(new_wait)
            return get_user_comments(author, new_wait)
    except requests.exceptions.ReadTimeout:
        print(f" Timed out. Trying again after {round(new_wait, 1)} seconds.")
        time.sleep(new_wait)
        return get_user_comments(author, new_wait)


if __name__ == "__main__":
    user_set = load_file(user_set_file)
    glob = load_file(glob_file)
    processed_users = load_file(processed_users_file)
    oldest_time = time.time()
    min_wait = .5

    if user_set:
        oldest_time = min([a['created_utc'] for a in user_set.values()])
    else:
        user_set = dict()
    if not glob:
        glob = {'text': [], 'scores': [], 'labels': []}
    if not processed_users:
        processed_users = set()

    raw_comments = []
    if len(user_set) < num_users:
        print("Beginning initial query for comments.")
        raw_comments = get_subreddit_data(oldest_time, 0)
    new_users = 0
    queried_comments = 0

    while len(raw_comments) > 0 and len(user_set) < num_users:
        init_time = time.time()
        print(f" Response received, adding users from {len(raw_comments)} comments from before "
              f"{datetime.fromtimestamp(oldest_time)}, currently have {len(user_set)}...", end='')
        for comment in raw_comments:
            queried_comments += 1
            if comment['author'] not in user_set:
                user_set[comment['author']] = {'created_utc': comment['created_utc'],
                                               'id': comment['id'],
                                               'author_flair_text': fmt(comment['author_flair_text'])}
                new_users += 1
        print(" Done!")
        if new_users > 100:
            write_file(user_set, user_set_file)
            queried_comments = 0
            new_users = 0
        oldest_time = raw_comments[-1]['created_utc']
        cur_time = time.time() - init_time
        wait_time = min_wait - cur_time
        if cur_time < min_wait:
            time.sleep(wait_time)
        raw_comments = get_subreddit_data(oldest_time, 0)

    print(f"Finished with {len(user_set)} total users. Proceeding to comment collection.")

    new_users = 0
    queried_users = 0
    init_time = time.time()
    for user in user_set:
        user_flair = user_set[user]['author_flair_text']
        if user_flair in flairs and user not in processed_users:
            cur_time = time.time() - init_time
            wait_time = min_wait - cur_time
            if cur_time < min_wait:
                time.sleep(wait_time)
            comments, flat = get_user_comments(user, 0)
            init_time = time.time()
            print(f" Response received, adding comment history, currently have {len(glob['text'])}...",
                  end='')
            if len(comments) > 0:
                glob['text'].append(flat)
                glob['scores'].append(round(sum(c['score'] for c in comments) / len(comments), 1))
                glob['labels'].append(flairs.index(user_flair))
                processed_users.add(user)
                new_users += 1
            queried_users += 1
            print(f" Done!")
        if new_users > 100:
            write_file(glob, glob_file)
            new_users = 0
            queried_users = 0
    print(f"Finished with {len(glob['text'])} comment histories.")
