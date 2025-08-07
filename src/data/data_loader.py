import gzip
import pandas as pd
from src.config import config


class DataLoader:
    """A centralized class for loading and caching all project data."""

    def __init__(self):
        self._ratings = None
        self._movies = None
        self._links = None
        self._imdb_ratings = None
        self._tags = None
        self._imdb_crew = None
        self._imdb_principals = None
        self._imdb_names = None
        self._imdb_movies = None

    def _load_ratings(self):
        if self._ratings is None:
            self._ratings = pd.read_csv(config.paths.RATINGS_FILE)

    def _load_movies(self):
        if self._movies is None:
            self._movies = pd.read_csv(config.paths.MOVIES_FILE)

    def _load_links(self):
        if self._links is None:
            links_df = pd.read_csv(config.paths.LINKS_FILE, dtype={'imdbId': str})
            links_df['tconst'] = 'tt' + links_df['imdbId'].str.zfill(7)
            self._links = links_df

    def _load_tags(self):
        if self._tags is None:
            self._tags = pd.read_csv(config.paths.TAGS_FILE)

    def _load_imdb_ratings(self):
        if self._imdb_ratings is None:
            self._load_links()
            relevant_tconsts = set(self.get_links()['tconst'])
            imdb_ratings_df = pd.read_table(config.paths.IMDB_RATINGS_FILE, compression='gzip', na_values=['\\N'])
            self._imdb_ratings = imdb_ratings_df[imdb_ratings_df['tconst'].isin(relevant_tconsts)].copy()

    def _load_imdb_movies(self):
        if self._imdb_movies is None:
            self._load_links()
            relevant_tconsts = set(self.get_links()['tconst'])
            usecols = ['tconst', 'isAdult', 'startYear', 'runtimeMinutes']
            imdb_movies_df = pd.read_table(config.paths.IMDB_MOVIES_FILE,
                                           compression='gzip', na_values=['\\N'],
                                           low_memory=False, usecols=usecols)
            self._imdb_movies = imdb_movies_df[imdb_movies_df['tconst'].isin(relevant_tconsts)].copy()

    def _load_imdb_crew(self):
        if self._imdb_crew is None:
            self._load_links()
            relevant_tconsts = set(self.get_links()['tconst'])

            crew_dtypes = {'tconst': 'string', 'directors': 'string'}
            crew_cols = ['tconst', 'directors']

            crew_df = pd.read_table(config.paths.IMDB_CREW_FILE,
                                    compression='gzip', na_values=['\\N'], usecols=crew_cols, dtype=crew_dtypes
                                    )

            self._imdb_crew = crew_df[crew_df['tconst'].isin(relevant_tconsts)].copy()
            self._imdb_crew.dropna(subset=['directors'], inplace=True)

    def _load_imdb_principals(self):
        if self._imdb_principals is None:
            self._load_links()
            relevant_tconsts = set(self.get_links()['tconst'])

            principals_dtypes = {
                'tconst': 'string',
                'ordering': 'uint8',
                'nconst': 'string',
                'category': 'category',
            }

            principals_cols = ['tconst', 'ordering', 'nconst', 'category']
            processed_chunks = []
            chunk_size = 1_000_000

            for chunk in pd.read_table(config.paths.IMDB_PRINCIPALS_FILE, compression='gzip',
                                       na_values=['\\N'], dtype=principals_dtypes, usecols=principals_cols,
                                       chunksize=chunk_size,
                                       ):
                filtered_chunk = chunk[
                    chunk['tconst'].isin(relevant_tconsts) &
                    chunk['category'].isin(['actor', 'actress'])
                    ]

                if not filtered_chunk.empty:
                    processed_chunks.append(filtered_chunk)

            self._imdb_principals = pd.concat(processed_chunks, ignore_index=True)

    def _load_imdb_names(self):
        if self._imdb_names is None:
            # First, get the nconsts we use
            self._load_imdb_principals()
            self._load_imdb_crew()

            # Collect all relevant nconsts
            relevant_nconsts = set()

            # From principals (actors)
            if self._imdb_principals is not None:
                relevant_nconsts.update(self._imdb_principals['nconst'].unique())

            # From crew (directors)
            if self._imdb_crew is not None:
                director_nconsts = self._imdb_crew['directors'].str.split(',').explode().unique()
                relevant_nconsts.update(director_nconsts)

            # Now load only relevant names
            names_df = pd.read_table(
                config.paths.IMDB_NAMES_FILE,
                compression='gzip',
                na_values=['\\N'],
                usecols=['nconst', 'primaryName']  # Only need these columns
            )

            # Filter to only names we need
            self._imdb_names = names_df[names_df['nconst'].isin(relevant_nconsts)].copy()

    def get_ratings(self) -> pd.DataFrame:
        self._load_ratings()
        return self._ratings

    def get_movies(self) -> pd.DataFrame:
        self._load_movies()
        return self._movies

    def get_links(self) -> pd.DataFrame:
        self._load_links()
        return self._links

    def get_imdb_ratings(self) -> pd.DataFrame:
        self._load_imdb_ratings()
        return self._imdb_ratings

    def get_tags(self) -> pd.DataFrame:
        self._load_tags()
        return self._tags

    def get_imdb_crew(self) -> pd.DataFrame:
        self._load_imdb_crew()
        return self._imdb_crew

    def get_imdb_names(self) -> pd.DataFrame:
        self._load_imdb_names()
        return self._imdb_names

    def get_imdb_movies(self) -> pd.DataFrame:
        self._load_imdb_movies()
        return self._imdb_movies

    def get_imdb_principals(self) -> pd.DataFrame:
        self._load_imdb_principals()
        return self._imdb_principals
