import requests
import json
import pickle
import time
import string

data_file = "Data/data.pickle"
glob_file = "Data/glob.pickle"
size = 250


try:
    with open(data_file, 'rb') as fp:
        dataSet = pickle.load(fp)
        oldestTS = min(a['created_utc'] for a in dataSet.values())
except FileNotFoundError as error:
    oldestTS = int(time.time())
    dataSet = dict()


def fmt(flair):
    if flair is not None:
        flair = flair.split(':')[1].lower()
        if flair[0] == 'c':
            return 'centrist'
        if flair[-1] == '2':
            return 'libright'
        return flair
    else:
        return None


def get_subreddit_data(a):
    url = 'https://api.pushshift.io/reddit/comment/search/?size='+str(size)+'&before='+str(a) + \
          '&subreddit=PoliticalCompassMemes&filter=id,author,author_flair_text,created_utc'
    r = requests.get(url)
    if r:
        data = json.loads(r.text)
        return data['data']
    else:
        time.sleep(3)
        return get_subreddit_data(a)


def get_user_comments(author):
    url = 'https://api.pushshift.io/reddit/comment/search/?size=50&author={}&filter=body,score'.format(author)
    r = requests.get(url)
    if r:
        comments = json.loads(r.text)['data']
        cleaned = []
        for comment in comments:
            if 'https://' not in comment['body'] and 'http://' not in comment['body']:
                new_body = ''.join(filter(lambda x: x in string.printable and x not in ',./?";\'`:;!@#$%^&*()_+1234567890-=\\|}{[]',
                                          comment['body'])).replace('\n', ' ')
                cleaned.append({'body': new_body,
                                'score': comment['score']})

        flat = ' '.join([c['body'] for c in cleaned])

        return cleaned, flat
    else:
        print(r)
        time.sleep(4)
        return get_user_comments(author)

before = oldestTS
data = get_subreddit_data(before)
cache_flag = 0

while len(data) >= 0 and len(dataSet) < 30000:
    new_users = 0
    for c in data:
        if c['author'] not in dataSet:
            dataSet[c['author']] = {'created_utc': c['created_utc'],
                                    'id': c['id'],
                                    'author_flair_text': fmt(c['author_flair_text'])}
            new_users += 1

    cache_flag += new_users
    if cache_flag >= 1000:
        with open(data_file, 'wb') as fp:
            pickle.dump(dataSet, fp)
        cache_flag = 0
        print("Cached dataset.")

    print('Got {total} users, added {new_users} users for a total of {full} users.'.format(total=len(data), new_users=new_users, full=len(dataSet)))
    data = get_subreddit_data(data[-1]['created_utc'])

score_tensor = []
text_tensor = []
labels = []

flairs = ['left', 'right', 'lib', 'auth', 'centrist', 'authright', 'libright', 'libleft', 'authleft']
cache_flag = 0

for user in dataSet:
    user_flair = dataSet[user]['author_flair_text']
    if user_flair in flairs:
        comments, flat = get_user_comments(user)
        if len(comments) > 0:
            text_tensor.append(flat)
            score_tensor.append(round(sum(c['score'] for c in comments)/len(comments), 1))
            labels.append(flairs.index(user_flair))
            print(flairs.index(user_flair))
        cache_flag += 1
        if cache_flag >= 50:
            glob = {'text': text_tensor, 'scores': score_tensor, 'labels': labels}
            with open(glob_file, 'wb') as fp:
                pickle.dump(glob, fp)
            cache_flag = 0
            print(len(score_tensor), len(text_tensor))

glob = {'text': text_tensor, 'scores': score_tensor, 'labels': labels}
with open(glob_file, 'wb') as fp:
    pickle.dump(glob, fp)
