import os
import pandas as pd


class SpotifyAnalysis:

    def __init__(self, spotify_data):
        self.profiles = dict()
        for userName in os.listdir(spotify_data):
            profile_directory = os.path.join(spotify_data, userName)
            print(userName, end=": ")
            self.profiles[userName] = pd.DataFrame()
            for streamHistory in os.listdir(profile_directory):
                if streamHistory.startswith("Streaming_History_Audio_") and streamHistory.endswith(".json"):
                    self.profiles[userName] = \
                        pd.concat(
                            [self.profiles[userName], pd.read_json(os.path.join(profile_directory, streamHistory))])

            self.profiles[userName].rename(columns={'ms_played': 'ms', 'master_metadata_track_name': 'track',
                                                    'master_metadata_album_artist_name': 'artist',
                                                    'master_metadata_album_album_name': 'album',
                                                    'spotify_track_uri': 'uri'}, inplace=True)
            self.profiles[userName] = self.profiles[userName][
                ['ts', 'ms', 'track', 'artist', 'album', 'uri', 'reason_start', 'reason_end', 'shuffle']]
            self.profiles[userName] = self.profiles[userName][self.profiles[userName]['track'].notna()]
            self.profiles[userName]['freq'] = 1
            self.profiles[userName]['skipped'] = (self.profiles[userName]['reason_end'] != "trackdone").astype(int)
            self.profiles[userName]["ts"] = pd.to_datetime(self.profiles[userName]["ts"])
            self.profiles[userName].set_index('ts', inplace=True)
            self.profiles[userName] = self.profiles[userName].sort_index()
            print(str(len(self.profiles[userName])) + " streams")

    def get_users(self):
        return list(self.profiles.keys())

    def get_profile(self, username=None):
        if not username:
            username = self.get_users()[0]
        return self.profiles[username]

    def get_years(self, username):
        return self.profiles[username].index.year.unique()

    def get_size(self, username):
        return len(self.profiles[username])