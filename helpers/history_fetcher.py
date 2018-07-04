import requests
import numpy as np
import dateutil.parser

WIKI_URL = 'https://en.wikipedia.org/w/api.php'

class HistoryFetcher:

    def __init__(self, wikipage_title):
        self.wikipage_title = wikipage_title
        self.payload = {'action': 'query',
                        'prop': 'revisions',
                        'titles': wikipage_title,
                        'rvprop': 'ids|flags|timestamp|comment|user|size',
                        'format': 'json',
                        'rvlimit': 500}

    def get_history(self, start_date, end_date):
        req_payload = dict(self.payload)

        # Start and end dates are inverted here as I find wikipedia ways of
        # thinking about edits history not intuitive
        req_payload['rvstart'] = end_date
        req_payload['rvend'] = start_date

        response = {'continue': True}
        revisions = []

        while 'continue' in response:
            response = requests.get(WIKI_URL, params=req_payload).json()
            response_data = list(response['query']['pages'].values())[0]['revisions']
            revisions.extend(response_data)

            if 'continue' in response:
                req_payload['rvcontinue'] = response['continue']['rvcontinue']

        head_payload = dict(self.payload)
        head_payload['rvlimit'] = 1
        head_payload['rvstart'] = start_date

        head_response = requests.get(WIKI_URL, params=head_payload).json()
        head_response_data = list(head_response['query']['pages'].values())[0]['revisions']

        revisions.extend(head_response_data)

        changes_size = self.__add_change_size(revisions)

        for i, revision in enumerate(revisions):
            revision['change_size'] = changes_size[i]

        for revision in revisions:
            revision['timestamp'] = dateutil.parser.parse(revision['timestamp'])

        revisions.pop()

        return revisions

    def get_edits_dates(self, start_date, end_date):
        revisions = self.get_history(start_date, end_date)
        return [edit['timestamp'] for edit in revisions]

    def __add_change_size(self, revisions):
        sizes = np.array(list(map(lambda revision: revision['size'], revisions)))

        change_size = np.insert(sizes, 0, 0) - np.append(sizes, [0])
        change_size = change_size[1:]
        change_size[-1:] = 0

        return change_size
